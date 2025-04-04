"""
This module offers an implementation of Equivalent Legendre polynomials [1]_.

Notes
-----
.. [1] Abedian, Alireza, and Alexander Düster.
   "Equivalent Legendre polynomials: Numerical integration of discontinuous functions in the finite element methods."
   Computer Methods in Applied Mechanics and Engineering 343 (2019): 690-720.
"""

from typing import Any, TypeAlias, Sequence, Callable
from dataclasses import dataclass

from jax.tree_util import register_pytree_node_class

from .prelude import *
from .r_fun import ADF
from .utils import apply_along_last_dims
from .integrate import make_quad_rule, gauss, Weights, Nodes


Domain: TypeAlias = Array | Sequence[Array]
LegendreCoefs: TypeAlias = Array
Scalar: TypeAlias = Array
_Moments: TypeAlias = Array


@register_pytree_node_class
@dataclass(frozen=True, slots=True, weakref_slot=True)
class ELP:
    domain: Domain
    coefs: LegendreCoefs

    def __call__(self, x: Array) -> Array:
        return _elp(x, self.coefs, self.domain)

    def tree_flatten(self):
        children = (self.domain, self.coefs)  # arrays / dynamic values
        return (children, None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children[0], children[1])


@partial(jit, static_argnames="n")
def legendre_polynomial(x: Array, n: int) -> Array:
    """Returns the first `n` Legendre polynomials evaluated at `x`.

    Parameters
    ----------
    x : Array
    n : int

    Returns
    -------
    Array
    """
    if n <= 0:
        raise ValueError("Degree must be greater than 0.")

    p = [jnp.ones_like(x), x]
    for deg in range(1, n):
        pn = ((2 * deg + 1) * x * p[deg] - deg * p[deg - 1]) / (deg + 1)
        p.append(pn)

    pn = asarray(p)[:n]
    return jnp.moveaxis(pn, 0, -1)


@partial(jit, static_argnames="n")
def legendre_poly_antiderivative(x: Array, n: int) -> Array:
    """Returns the first `n` antiderivatives of the Legendre polynomials evaluated at `x`.

    Parameters
    ----------
    x : Array
    n : int

    Returns
    -------
    Array
    """
    if n > 1:
        p2, p1 = legendre_polynomial(x, n), legendre_polynomial(x, n - 1)
        p = p2.at[..., 1:].set(x[..., None] * p2[..., 1:] - p1)
    else:
        p = legendre_polynomial(x, n)
    p = p.at[..., 0].set(x * p[..., 0])
    return p / jnp.arange(1, n + 1)


@partial(jit, static_argnames=("adf", "degree", "max_depth"))
def compute_elp(
    adf: ADF,
    domain: Domain,
    degree: int | Sequence[int],
    *args: Any,
    eps: float = 1e-6,
    max_depth: int = 3,
    **kwargs: Any,
) -> ELP:
    """Computes the Equivalent Legendre Polynomial for a given Approximate Distance Function.

    Parameters
    ----------
    adf : ADF
    domain : Domain
    degree : int | tuple[int]
    args : Any
        args for `adf`
    eps : float, optional
        controlls the precision whether a cell is contained by the `adf`, by default 1e-6
    max_depth : int, optional
        maximum depth of the space tree, by default 6
    kwargs : Any
        kwargs for `adf`
    Returns
    -------
    ELP
    """
    domain = _domain_grid(domain)
    if isinstance(degree, int):
        d = domain.shape[-1]
        degree = [degree] * d
        
    def _compute_coefs(lb, ub):
        dom = [array([_l, _u]) for _l, _u in zip(lb, ub)]
        dom = _domain_grid(dom)
        m = _compute_moments(
            adf, degree, dom, lb, ub,
            *args,
            eps=eps, max_depth=max_depth, **kwargs
        )
        c = [(2 * jnp.arange(_degree) + 1) / 2 for _degree in degree]
        c = jnp.prod(jnp.stack(jnp.meshgrid(*c, indexing="ij"), axis=-1), axis=-1)
        return c * m
    
    coefs = _apply_on_domain(_compute_coefs, domain)
    return ELP(domain, coefs)


@jit
def make_elp_quad_rule(
    elp: ELP,
) -> tuple[Weights, Nodes]:
    """Computes a quadrature rule for the given ELP.

    Parameters
    ----------
    elp : ELP

    Returns
    -------
    tuple[Weights, Nodes]
    """
    def compute_on_node(coefs, weight, node):
        leg_poly = [legendre_polynomial(x, p) for x, p in zip(node, coefs.shape)]
        elp = jnp.sum(coefs * jnp.prod(jnp.stack(jnp.meshgrid(*leg_poly, indexing="ij"), axis=-1), axis=-1))
        return elp * weight
    
    def _make_elp_quad_rule(lb, ub, coefs):
        dim = lb.shape[0]
        d = [jnp.array((_l, _u)) for _l, _u in zip(lb, ub)]
        degrees = coefs.shape
        weights, nodes = make_quad_rule(d, method=[gauss(_degree) for _degree in degrees])
        weights, nodes = weights[*[0] * dim], nodes[*[0] * dim]
        weights = apply_along_last_dims(lambda w, x: compute_on_node(coefs, w, center(x, lb, ub)), weights, nodes)
        return weights, nodes
    
    return _apply_on_domain(_make_elp_quad_rule, elp.domain, elp.coefs)


def _elp(x: Array, coefs: LegendreCoefs, domain: Domain) -> Scalar:
    def _leg_poly(x, n):
        L = legendre_polynomial(x, n)
        return asarray(jnp.where(((-1 < x) & (x <= 1))[..., None], L, 0))

    def _eval_elp(lb, ub, coefs):
        _x = center(x, lb, ub)
        leg_poly = [_leg_poly(xi, p) for xi, p in zip(_x, coefs.shape)]
        elp = jnp.sum(coefs * jnp.prod(jnp.stack(jnp.meshgrid(*leg_poly, indexing="ij"), axis=-1), axis=-1))
        return elp

    elp = jnp.sum(_apply_on_domain(_eval_elp, domain, coefs))
    return elp


_BrokenCellMask: TypeAlias = Array
_PaddingCellMask: TypeAlias = Array
_CellCenterMask: TypeAlias = Array
_FullCellMask: TypeAlias = Array


def _domain_masks(
    adf: ADF,
    domain: Domain,
    *args: Any,
    eps=1e-6,
    **kwargs: Any
) -> tuple[_FullCellMask, _BrokenCellMask, _PaddingCellMask, _CellCenterMask]:
    """For a given domain grid and a domain which is implicitly
    defined via the ADF, this function computes which cells are
    between the inside and outside of the domain (broken cells)
    and which cells are completely outside the domain (padding cells).

    Parameters
    ----------
    adf : ADF
    domain : Domain
        domain grid
    eps : _type_, optional
        threshold parameter for nodes inside and outside the boundary,
        e.g. cell with only one node inside the domain but `adf < eps` is
        still a padding cell, by default 1e-6

    Returns
    -------
    tuple[FullCellMask, _BrokenMask, _PaddingMask, _CenterMask]
    """
    domain = _domain_grid(domain)
    d = len(domain.shape) - 1  # dimension of the problem
    max_count = 2 ** d + 1
    
    def _make_mask(lb, ub):
        s = jnp.max(ub - lb)
        center = (lb + ub) / 2
        cell_dom = _domain_grid([array([_l, _u]) for _l, _u in zip(lb, ub)])
        ld = jnp.apply_along_axis(adf, -1, cell_dom, *args, **kwargs)
        lc = asarray(adf(center, *args, **kwargs))
        nd, nc = jnp.sum(ld >= (0 - eps)), (lc >= 0).astype(jnp.int32)
        count = nd + nc
        
        full_cell = count == max_count  # cell fully inside domain
        broken_cell = (count != max_count) & (count > 0)  # cell is broken by the domain boundary
        padding_cell = jnp.all(ld < (-s / 2)) & (lc < (-s / 2))  # cell is fully outside domain
        center_inside = lc >= 0
        return full_cell, broken_cell, padding_cell, center_inside
        
    return _apply_on_domain(_make_mask, domain)
    

def _compute_moments(
    adf: ADF,
    degree: int | Sequence[int],
    domain: Array,
    lower_bound: Array,
    upper_bound: Array,
    *args: Any,
    eps: float,
    max_depth: int,
    **kwargs: Any
) -> _Moments:
    """Computes moments and coefficients for the given domain for the
    equivalent legendre polynomials. The integration is performed with
    an adapted version of the recursive spacetrees algorithm from [1]_.
    The moments are evaluated for each cell in the domain grid.

    Parameters
    ----------
    adf : ADF
        describes the inside of the domain which is integrated
    degree : int | Sequence[int]
        the number of Legendre polynomials for each dimension, the polynomial degree is `degree + 1`
    domain : Array
        domain cell grid
    lower_bound : Array
        lower bound for the shifted legendre polynomial
    upper_bound : Array
        upper bound for the shifted legendre polynomial
    args : Any
        positional arguments for `adf`
    eps : float
        threshold parameter for domain masks
    max_depth : int
        maximum depth of the spacetree
    kwargs : Any
        keyword Arguments for `adf`
    Returns
    -------
    tuple[_Moments]
    """
    moments = _integrate_legendre_cell(adf, degree, domain, lower_bound, upper_bound, *args,
                                       eps=eps, max_depth=max_depth, depth=0,
                                       **kwargs)
    return moments
    

def _integrate_legendre_cell(
    adf: ADF,
    degree: int | Sequence[int],
    cell_domain: Array,
    lb: Array,
    ub: Array,
    *args: Any,
    eps: float,
    depth: int,
    max_depth: int,
    **kwargs: Any
) -> Array:
    # instead of binary split at the center, each cell is split into (splits ** dim) new cells
    # this increases convergece, lowers compile time and increases runtime.
    splits = 3
    d = cell_domain.ndim - 1
    if isinstance(degree, int):
        degree = [degree] * d
    
    full_cell_mask, broken_cell_mask, padding_cell_mask, cell_center_mask = _domain_masks(
        adf, cell_domain, *args, eps=eps, **kwargs)

    def _recursive_call(idx):
        Dl = cell_domain[*idx]
        Du = cell_domain[*[i + 1 for i in idx]]
        padding = padding_cell_mask[*idx]
        broken = broken_cell_mask[*idx]
        full_cell = full_cell_mask[*idx]
        center_in_domain = cell_center_mask[*idx]
        
        def _integrate(_l, _u):
            Pl = [legendre_poly_antiderivative(center(xi, li, ui), _d) for xi, li, ui, _d in zip(_l, lb, ub, degree)]
            Pu = [legendre_poly_antiderivative(center(xi, li, ui), _d) for xi, li, ui, _d in zip(_u, lb, ub, degree)]
            P = [Pui - Pli for Pui, Pli in zip(Pu, Pl)]
            IP = jnp.prod(jnp.stack(jnp.meshgrid(*P, indexing="ij"), axis=-1), axis=-1)
            return IP

        if depth == max_depth:
            IP = _integrate(Dl, Du)
            return lax.cond(
                full_cell,
                lambda: IP,
                lambda: lax.cond(broken,
                                 lambda: jnp.where(center_in_domain, IP, zeros_like(IP)),
                                 lambda: zeros_like(IP))
            )
            
        else:
            def _split():
                new_domain = [jnp.linspace(_l, _u, splits + 1) for _l, _u in zip(Dl, Du)]
                new_domain = jnp.stack(jnp.meshgrid(*new_domain, indexing="ij"), axis=-1)
                return _integrate_legendre_cell(
                    adf, degree, new_domain, lb, ub, *args, eps=eps, depth=depth + 1,
                    max_depth=max_depth, **kwargs)
            
            return lax.cond(
                full_cell,
                lambda: _integrate(Dl, Du),
                lambda: lax.cond(padding,
                                 lambda: zeros(degree),
                                 _split)
            )

    indices = jnp.indices(broken_cell_mask.shape)
    indices = jnp.moveaxis(indices, 0, -1)
    indices = indices.reshape(-1, d)
    return jnp.sum(lax.map(_recursive_call, indices, batch_size=splits), axis=0)


def shift(x, lb, ub, lb_new, ub_new):
    """Shift from (lb, ub) -> (lb_new, ub_new)

    Parameters
    ----------
    x : Array | float
    lb : Array | float
    ub : Array | float
    lb_new : Array | float
    ub_new : Array | float

    Returns
    -------
    Array | float
    """
    k = (ub_new - lb_new) / (ub - lb)
    return x * k - k * lb + lb_new


def center(x, lb, ub):
    """Centers the input; (lb, ub) -> (-1, 1)

    Parameters
    ----------
    x : Array | float
    lb : Array | float
    ub : Array | float

    Returns
    -------
    Array | float
    """
    return shift(x, lb, ub, -1, 1)


def _domain_grid(domain: Array | Sequence[Array]) -> Array:
    if not isinstance(domain, Array):
        domain = jnp.stack(jnp.meshgrid(*domain, indexing="ij"), axis=-1)
    else:
        domain = domain

    return domain


def _apply_on_domain(fn: Callable, domain: Domain, *args):
    domain = _domain_grid(domain)
    d = domain.shape[-1]

    def apply_fn(i):
        Di = lax.dynamic_slice(domain, jnp.concatenate([i, array([0])]), (2,) * d + (d,))
        lb = Di[*([0] * d)]
        ub = Di[*([-1] * d)]
        _args = [tree.map(lambda a: a[*i], a) for a in args]
        return fn(lb, ub, *_args)

    return _apply_on_indices(apply_fn, tuple(dim - 1 for dim in domain.shape[:-1]))


def _apply_on_indices(fn, shape):
    indices = jnp.indices(shape)
    indices = jnp.moveaxis(indices, 0, -1)
    return apply_along_last_dims(fn, indices)
