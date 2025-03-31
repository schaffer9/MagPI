"""
This module offers an implementation of Equivalent Legendre polynomials [1]_.

Notes
-----
.. [1] Abedian, Alireza, and Alexander Düster.
   "Equivalent Legendre polynomials: Numerical integration of discontinuous functions in the finite element methods."
   Computer Methods in Applied Mechanics and Engineering 343 (2019): 690-720.
"""

from typing import Any, NamedTuple, TypeAlias, Sequence, Callable
from dataclasses import dataclass
import itertools
import warnings

from jax.tree_util import register_pytree_node_class

from .prelude import *
from .r_fun import ADF, newton_iteration
from .utils import apply_along_last_dims
from .integrate import make_quad_rule, gauss, Weights, Nodes


_BrokenCellMask: TypeAlias = Array
_PaddingMask: TypeAlias = Array
Domain: TypeAlias = Array | Sequence[Array]
Coefs: TypeAlias = Array
LegendreCoefs: TypeAlias = Array
Scalar: TypeAlias = Array


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


class Cell(NamedTuple):
    lower_bound: Array
    upper_bound: Array
    broken: Array
    padding: Array

    def cell_count(self):
        return len(self.broken)

    def support(self):
        return jnp.prod((self.upper_bound - self.lower_bound), axis=-1)


Cells: TypeAlias = Cell


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


@partial(jit, static_argnames=("adf", "degree", "max_cells", "split_mode"))
def compute_elp(
    adf: ADF,
    domain: Domain,
    degree: int | tuple[int],
    *args: Any,
    max_cells: int = 100_000,
    eps: float = 1e-6,
    max_depth: int = 6,
    split_mode: str = "boundary",
    newton_maxiter: int = 10,
    **kwargs: Any,
) -> ELP:
    """Computes the Equivalent Legendre Polynomial for a given Approximate Distance Function.

    Parameters
    ----------
    adf : ADF
    domain : Domain
    degree : int | tuple[int]
    max_cells : int, optional
        maximum number of cells; if less cells are required, the remaining cells are only for
        padding; by default 100_000
    eps : float, optional
        controlls the precision whether a cell is contained by the `adf`, by default 1e-6
    max_depth : int, optional
        maximum depth of the space tree, by default 6
    split_mode : str, optional
        split either on the "boundary" or the "center" for each cell, by default "boundary"
    newton_maxiter : int, optional
        maximum newton iterations to find a point on the boundary, by default 10

    Returns
    -------
    ELP
    """
    domain = _domain_grid(domain)

    cells = partition_domain(
        adf,
        domain,
        *args,
        max_cells=max_cells,
        eps=eps,
        max_depth=max_depth,
        split_mode=split_mode,
        newton_maxiter=newton_maxiter,
        **kwargs,
    )
    
    def _coefs(lb, ub, cells):
        return _compute_coefs(cells, lb, ub, degree)
    
    coefs = _apply_on_domain(_coefs, domain, cells)
    return ELP(domain, coefs)


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
        d = [jnp.array((l, u)) for l, u in zip(lb, ub)]
        degrees = coefs.shape
        weights, nodes = make_quad_rule(d, method=[gauss(_degree) for _degree in degrees])
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


def partition_domain(
    adf: ADF,
    domain: Domain,
    *args: Any,
    max_cells: int = 100_000,
    eps: float = 1e-6,
    max_depth: int = 6,
    split_mode: str = "boundary",
    newton_maxiter: int = 10,
    **kwargs: Any,
) -> Cells:
    """Partitions the computational domain into cuboid cells

    Parameters
    ----------
    adf : ADF
    domain : Domain
    max_cells : int, optional
        maximum number of cells; if less cells are required, the remaining cells are only for
        padding; by default 100_000
    eps : float, optional
        controlls the precision whether a cell is contained by the `adf`, by default 1e-6
    max_depth : int, optional
        maximum depth of the space tree, by default 6
    split_mode : str, optional
        split either on the "boundary" or the "center" for each cell, by default "boundary"
    newton_maxiter : int, optional
        maximum newton iterations to find a point on the boundary, by default 10

    Returns
    -------
    Cells
    """
    D = _domain_grid(domain)

    d = D.ndim - 1
    lb = D[*[slice(0, -1) for _ in range(d)]]
    ub = D[*[slice(1, None) for _ in range(d)]]

    def _partition(lb, ub):
        return _partition_domain(
            adf,
            lb,
            ub,
            *args,
            eps=eps,
            max_depth=max_depth,
            max_cells=max_cells,
            split_mode=split_mode,
            newton_maxiter=newton_maxiter,
            **kwargs,
        )

    return apply_along_last_dims(_partition, lb, ub)


@partial(jit, static_argnames=("adf", "max_cells", "split_mode"))
def _partition_domain(
    adf: ADF,
    lower_bounds: Array,
    upper_bounds: Array,
    *args: Any,
    eps: float = 1e-6,
    max_depth: int = 8,
    max_cells: int = 100_000,
    split_mode: str = "boundary",
    newton_maxiter: int = 10,
    **kwargs: Any,
) -> Cells:
    lb, ub = asarray(lower_bounds), asarray(upper_bounds)
    lower_bounds = zeros((max_cells, lb.shape[0]))
    upper_bounds = zeros((max_cells, ub.shape[0]))
    broken_mask = jnp.full((max_cells,), False)
    padding_mask = jnp.full((max_cells,), True)
    d = lb.shape[0]
    n = 2**d
    assert lower_bounds.shape == upper_bounds.shape

    lower_bounds = lower_bounds.at[0].set(lb)
    upper_bounds = upper_bounds.at[0].set(ub)

    is_broken_cell, is_padding_cell = _broken_or_padding_cell(adf, lb, ub, *args, eps=eps, **kwargs)

    broken_mask = broken_mask.at[0].set(is_broken_cell)
    padding_mask = padding_mask.at[0].set(is_padding_cell)
    cells = Cell(lower_bounds, upper_bounds, broken_mask, padding_mask)

    def split(cell):
        lb, ub, broken, padding = cell

        def _no_split():
            lower_bounds = zeros((n, lb.shape[0]))
            lower_bounds = lower_bounds.at[0].set(lb)
            upper_bounds = zeros((n, ub.shape[0]))
            upper_bounds = upper_bounds.at[0].set(ub)
            broken_mask = jnp.full((n,), False)
            broken_mask = broken_mask.at[0].set(broken)
            padding_mask = jnp.full((n,), True)
            padding_mask = padding_mask.at[0].set(padding)
            new_cells = Cell(lower_bounds, upper_bounds, broken_mask, padding_mask)
            return new_cells

        def _split():
            new_cells = _split_cell(
                adf, lb, ub, *args, eps=eps, split_mode=split_mode, tol=eps / 2, maxiter=newton_maxiter, **kwargs
            )
            return new_cells

        v = jnp.prod(cell.upper_bound - cell.lower_bound)
        not_too_small_to_split = ~(v < (eps / (10 ** (d - 1))))  # avoid splitting cells which are already very small
        return lax.cond(broken & jnp.logical_not(padding) & not_too_small_to_split, _split, _no_split)

    def split_cells(i, cells):
        new_cells = vmap(split)(cells)
        new_cells = new_cells._replace(
            lower_bound=new_cells.lower_bound.reshape(-1, d),
            upper_bound=new_cells.upper_bound.reshape(-1, d),
            broken=new_cells.broken.reshape(-1),
            padding=new_cells.padding.reshape(-1),
        )

        mask = new_cells.padding
        new_cell_count = jnp.count_nonzero(jnp.logical_not(mask))

        if split_mode == "center":
            # if the maximum depth is reached, cells with the center outside the domain
            # are removed. This increases the accuracy.
            mask = lax.cond(
                (i == (max_depth - 1)) | (new_cell_count > max_cells),
                lambda: vmap(lambda c: jnp.logical_not(_center_inside_adf(adf, c, *args, **kwargs)))(new_cells) | mask,
                lambda: mask,
            )
        new_cells = new_cells._replace(padding=mask)
        new_cell_count = jnp.count_nonzero(jnp.logical_not(mask))
        cell_overflow = new_cell_count > max_cells
        jax.debug.callback(_warn_on_overflow, max_cells, new_cell_count, i + 1)
        idx = jnp.argsort(mask)
        idx = idx[:max_cells]

        _cells = cells._replace(
            lower_bound=new_cells.lower_bound[idx],
            upper_bound=new_cells.upper_bound[idx],
            broken=new_cells.broken[idx],
            padding=new_cells.padding[idx],
        )
        return lax.cond(cell_overflow, lambda: cells, lambda: _cells), cell_overflow

    def cond_fun(state):
        depth, overflow, _ = state
        resume = (depth < max_depth) & (~overflow)
        return resume

    def body(state):
        depth, _, cells = state
        cells, cell_overflow = split_cells(depth, cells)
        return depth + 1, cell_overflow, cells

    _, _, cells = lax.while_loop(cond_fun, body, (0, False, cells))
    cells = vmap(_pad_cell)(cells)  # set all padding cells to zero
    if split_mode == "boundary":
        # in this mode, the broken cells are halved along the shorter cell dimension
        # to have higher accuracy
        cells = _halve_broken_cells(adf, cells, *args, **kwargs)
    return cells


def _warn_on_overflow(max_cells, new_cell_count, depth):
    if new_cell_count > max_cells:
        warnings.warn(
            f"`max_cells`={max_cells} is not big enough at depth {depth}. There are {new_cell_count} cells. Iteration stopped!"
        )


def _split_broken_cell(adf, cell, *args, **kwargs):
    split = cell.broken & (~cell.padding)
    lb, ub = cell.lower_bound, cell.upper_bound
    s = ub - lb
    i = jnp.argmin(s)
    c1 = Cell(lb, ub.at[i].set(ub[i] - s[i] / 2), cell.broken, cell.padding)
    c2 = Cell(lb.at[i].set(lb[i] + s[i] / 2), ub, cell.broken, cell.padding)
    c_adf1 = adf((c1.lower_bound + c1.upper_bound) / 2, *args, **kwargs)
    c_adf2 = adf((c2.lower_bound + c2.upper_bound) / 2, *args, **kwargs)
    return lax.cond(split, lambda: lax.cond(c_adf1 > c_adf2, lambda: c1, lambda: c2), lambda: cell)


def _halve_broken_cells(adf, cells, *args, **kwargs):
    return apply_along_last_dims(lambda c: _split_broken_cell(adf, c, *args, **kwargs), cells)


def _pad_cell(cell):
    lb, ub, _, padding_cell = cell
    return lax.cond(
        padding_cell,
        lambda: Cell(
            lower_bound=zeros_like(lb),
            upper_bound=zeros_like(ub),
            broken=asarray(False),
            padding=asarray(True),
        ),
        lambda: cell,
    )


def _center_inside_adf(adf, cell, *args, **kwargs):
    lb, ub, _, _ = cell
    c = (lb + ub) / 2
    return adf(c, *args, **kwargs) >= 0


def _split_cell(adf: ADF, lb, ub, *args: Any, eps, split_mode, tol, maxiter, **kwargs: Any) -> Cell:
    c = split_point = (lb + ub) / 2
    if split_mode == "center":
        split_point = c
    elif split_mode == "boundary":
        split_point = newton_iteration(adf, c, *args, tol=tol, maxiter=maxiter, **kwargs)
        # if the split_point is outside the cell, then we take the center
        split_point = lax.cond(jnp.all((lb < split_point) & (split_point < ub)), lambda: split_point, lambda: c)
    else:
        raise ValueError("`split_mode` must be 'center' or 'boundary'")
    lower_bounds, upper_bounds, broken_cells, padding_cells = [], [], [], []
    for split in itertools.product([0, 1], repeat=len(lb)):
        new_lower = where(array(split) == 0, lb, split_point)
        new_upper = where(array(split) == 0, split_point, ub)
        broken_cell, padding_cell = _broken_or_padding_cell(adf, new_lower, new_upper, *args, eps=eps, **kwargs)
        lower_bounds.append(new_lower)
        upper_bounds.append(new_upper)
        broken_cells.append(broken_cell)
        padding_cells.append(padding_cell)

    _lower_bounds = asarray(lower_bounds)
    _upper_bounds = asarray(upper_bounds)
    _broken_cells = asarray(broken_cells)
    _padding_cells = asarray(padding_cells)
    return Cell(_lower_bounds, _upper_bounds, _broken_cells, _padding_cells)


@partial(jit, static_argnames=("adf",))
def _broken_or_padding_cell(
    adf: ADF, lb, ub, *args: Any, eps=1e-6, **kwargs: Any
) -> tuple[_BrokenCellMask, _PaddingMask]:
    support = jnp.prod((ub - lb) / 2)
    assert lb.shape[0] == ub.shape[0]
    dim = lb.shape[0]
    cell_domain = jnp.stack(jnp.meshgrid(*[jnp.array([l, u]) for l, u in zip(lb, ub)], indexing="ij"), axis=-1)
    center = (lb + ub) / 2

    LD = jnp.apply_along_axis(adf, -1, cell_domain, *args, **kwargs)
    LX = adf(center, *args, **kwargs)
    max_count = 1 + 2**dim
    nD = jnp.sum(LD >= 0 - eps)
    nX = asarray((LX >= (0 - eps))).astype(nD.dtype)
    n = nD + nX
    padding_cell = ((n == 0) | ((nX == 0) & jnp.all(LD <= 0 + eps))) | (support == 0)
    broken_cell = (n != max_count) & (n > 0)
    broken_cell = asarray(jnp.where(padding_cell, False, broken_cell))
    return broken_cell, padding_cell


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


def _compute_coefs(cells, lb, ub, degree):
    d = cells.lower_bound.shape[-1]
    
    if isinstance(degree, int):
        degree = [degree] * d

    def _integrate_1d(lb, ub, degree):
        l_lb = legendre_poly_antiderivative(lb, degree + 1)
        l_ub = legendre_poly_antiderivative(ub, degree + 1)
        a = l_ub - l_lb
        return a
    
    def moment(c):
        _lb = center(c.lower_bound, lb, ub)
        _ub = center(c.upper_bound, lb, ub)
        a = [_integrate_1d(l, u, d) for l, u, d in zip(_lb, _ub, degree)]    
        return jnp.prod(jnp.stack(jnp.meshgrid(*a, indexing="ij"), axis=-1), axis=-1)

    def body(carry, c):
        m = moment(c)
        return jnp.add(carry, m), None

    c = [(2 * jnp.arange(_degree + 1) + 1) / 2 for _degree in degree]
    c = jnp.prod(jnp.stack(jnp.meshgrid(*c, indexing="ij"), axis=-1), axis=-1)
    carry = zeros_like(c)
    m, _ = lax.scan(body, carry, cells)
    return c * m


def _domain_grid(domain: Domain) -> Array:
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
