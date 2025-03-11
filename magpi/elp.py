"""
This module offers an implementation of Equivalent Legendre polynomials [1]_.

Notes
-----
.. [1] Abedian, Alireza, and Alexander Düster.
   "Equivalent Legendre polynomials: Numerical integration of discontinuous functions in the finite element methods." 
   Computer Methods in Applied Mechanics and Engineering 343 (2019): 690-720.
"""

from typing import Any, NamedTuple, TypeAlias, Sequence
import itertools
import warnings

from .prelude import *
from .r_fun import ADF, newton_iteration
from .utils import apply_along_last_dims


_BrokenCellMask: TypeAlias = Array
_PaddingMask: TypeAlias = Array
Domain: TypeAlias = Array
Weights: TypeAlias = Array
Nodes: TypeAlias = Array
Moments: TypeAlias = Array
LegendreCoefs: TypeAlias = Array
Scalar: TypeAlias = Array


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


@partial(jit, static_argnames=("degree",))
def compute_coefs(cells: Cells, domain: Domain, degree: int):
    D = domain
    d = D.shape[-1]
    def _coefs(i):
        Di = lax.dynamic_slice(D, jnp.concatenate([i, array([0])]), (2,) * d + (d,))
        lb = Di[*([0] * d)]
        ub = Di[*([-1] * d)]
        return _compute_coefs(tree.map(lambda c: c[*i], cells), lb, ub, degree + 1)

    return _apply_on_indices(_coefs, tuple(dim - 1 for dim in D.shape[:-1]))


def compute_elp_weights(
    coefs: LegendreCoefs,
    weights: Weights,
    nodes: Nodes,
    domain: Domain,
) -> Weights:
    """Evaluates the Equivalent Legendre Polynomials for the given quadrature
    rule. Note that the domain must be the same which was used to evaluate the 
    ELP coefficients.

    Parameters
    ----------
    coefs : LegendreCoefs
    weights : Weights
        quadrature weights
    nodes : Nodes
        quadrature nodes
    domain : Domain
        domain grid

    Returns
    -------
    Weights
    """
    d = domain.ndim - 1
    lb = domain[*[slice(0, -1) for _ in range(d)]]
    ub = domain[*[slice(1, None) for _ in range(d)]]

    def compute_on_node(coefs, weight, node):
        leg_poly = [legendre_polynomial(x, p) for x, p in zip(node, coefs.shape)]
        elp = jnp.sum(coefs * jnp.prod(jnp.stack(jnp.meshgrid(*leg_poly), axis=-1), axis=-1))
        return elp * weight

    def compute_new_weights(coefs, weights, nodes, lb, ub):
        nodes = (nodes * 2 - (lb + ub)) / (ub - lb)
        new_weights = apply_along_last_dims(lambda w, x: compute_on_node(coefs, w, x), weights, nodes)
        return new_weights

    new_weights = apply_along_last_dims(compute_new_weights, coefs, weights, nodes, lb, ub, dims=d + 1)
    return new_weights

# TODO:
# def elp(x: Array, coefs: LegendreCoefs, domain: Domain) -> Scalar:

#     d = domain.ndim - 1
#     lb = domain[*[slice(0, -1) for _ in range(d)]]
#     ub = domain[*[slice(1, None) for _ in range(d)]]

#     def compute_on_node(coefs, weight, node):
#         inside = 
#         leg_poly = [legendre_polynomial(x, p) for x, p in zip(node, coefs.shape)]
#         elp = jnp.sum(coefs * jnp.prod(jnp.stack(jnp.meshgrid(*leg_poly), axis=-1), axis=-1))
#         return elp * weight
    
#     def compute_new_weights(coefs, lb, ub):
#         _x = (x * 2 - (lb + ub)) / (ub - lb)
#         new_weights = apply_along_last_dims(lambda w, x: compute_on_node(coefs, w, x), weights, nodes)
#         return new_weights

#     new_weights = apply_along_last_dims(compute_new_weights, coefs, weights, nodes, lb, ub, dims=d + 1)
#     return new_weights

def partition_domain(
    adf: ADF,
    domain: Sequence[Array],
    *args: Any,
    max_cells=100_000,
    eps=1e-6,
    max_depth=6,
    split_mode: str = "boundary",
    newton_maxiter: int = 10,
    **kwargs: Any
) -> Cells:
    D = jnp.stack(jnp.meshgrid(*domain), axis=-1)
    d = D.ndim - 1
    lb = D[*[slice(0, -1) for _ in range(d)]]
    ub = D[*[slice(1, None) for _ in range(d)]]

    def _partition(lb, ub):
        return _partition_domain(
            adf,
            lb, ub,
            *args,
            eps=eps,
            max_depth=max_depth,
            max_cells=max_cells,
            split_mode=split_mode,
            newton_maxiter=newton_maxiter,
            **kwargs
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
    """Partitions the domain contained by the approximate distance function `adf`
    and `lower_bounds` and `upper_bounds` into cells with a space tree algorithm.
    The resolution becomes smaller and smaller it the boundary of the domain is approached
    until the maximum resolution is reached. The maximum resolution is based on `max_depth`
    and `max_cells`.

    The algorithm refines the domain iteratively since a recursive approach would require
    huge amounts of memory in JAX.

    Parameters
    ----------
    adf : ADF
        approximate distance function
    lower_bounds : Array
    upper_bounds : Array
    eps : float, optional
        controlls the precision whether a cell is contained by the `adf`, by default 1e-6
    max_depth : int, optional
        maximum depth of the space tree, by default 8
    max_cells : int, optional
        maximum number of cells; if less cells are required, the remaining cells are only for
        padding; by default 100_000

    Returns
    -------
    Cells
        Domain partitioned into computational cells
    """

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
            new_cells = _split_cell(adf, lb, ub, *args,
                                    eps=eps, split_mode=split_mode,
                                    tol=eps/2, maxiter=newton_maxiter, **kwargs)
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
            f"`max_cells`={max_cells} is not big enough at depth {depth}. There are {new_cell_count} cells. Iteration stopped!")


def _split_broken_cell(adf, cell, *args, **kwargs):
    split = cell.broken & (~cell.padding)
    lb, ub = cell.lower_bound, cell.upper_bound
    s = ub - lb
    i = jnp.argmin(s)
    c1 = Cell(lb, ub.at[i].set(ub[i] - s[i] / 2), cell.broken, cell.padding)
    c2 = Cell(lb.at[i].set(lb[i] + s[i] / 2), ub, cell.broken, cell.padding)
    c_adf1 = adf((c1.lower_bound + c1.upper_bound) / 2, *args, **kwargs)
    c_adf2 = adf((c2.lower_bound + c2.upper_bound) / 2, *args, **kwargs)
    return lax.cond(split,
                    lambda: lax.cond(c_adf1 > c_adf2, lambda: c1, lambda: c2),
                    lambda: cell)


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
        split_point = lax.cond(
            jnp.all((lb < split_point) & (split_point < ub)),
            lambda: split_point,
            lambda: c
        )
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
def _broken_or_padding_cell(adf: ADF, lb, ub, *args: Any, eps=1e-6, **kwargs: Any) -> tuple[_BrokenCellMask, _PaddingMask]:
    support = jnp.prod((ub - lb) / 2)
    assert lb.shape[0] == ub.shape[0]
    dim = lb.shape[0]
    cell_domain = jnp.stack(jnp.meshgrid(*[jnp.array([l, u]) for l, u in zip(lb, ub)]), axis=-1)
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


def _compute_coefs(cells, lb, ub, degree):
    d = cells.lower_bound.shape[-1]

    def moment(c):
        _lb = (c.lower_bound * 2 - (lb + ub)) / (ub - lb)
        _ub = (c.upper_bound * 2 - (lb + ub)) / (ub - lb)
        l_lb = legendre_poly_antiderivative(_lb, degree)
        l_ub = legendre_poly_antiderivative(_ub, degree)
        a = l_ub - l_lb
        return jnp.prod(jnp.stack(jnp.meshgrid(*a), axis=-1), axis=-1)

    def body(carry, c):
        m = moment(c)
        return jnp.add(carry, m), None
    
    c = [(2 * jnp.arange(degree) + 1) / 2 for _ in range(d)]
    c = jnp.prod(jnp.stack(jnp.meshgrid(*c), axis=-1), axis=-1)
    carry = zeros_like(c)
    m, _ = lax.scan(body, carry, cells)
    return c * m


def _apply_on_indices(fn, shape):
    indices = jnp.indices(shape)
    indices = jnp.moveaxis(indices, 0, -1)
    return apply_along_last_dims(fn, indices)