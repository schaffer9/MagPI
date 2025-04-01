from magpi.prelude import *


def _base_fun(x: Array, k: int, i: int, t: Array, degree: int, extrapolate: bool) -> Array:
    if k == 0:
        n = len(t) - k - 1
        a1 = jnp.where((t[..., i] <= x) & (x < t[..., i + 1]), 1.0, 0.0)
        if extrapolate:
            # a2 and a3 for extrapolation
            a2 = jnp.where((x < t[..., 0]) & (i <= degree), 1.0, 0.0)
            a3 = jnp.where((x >= t[..., -1]) & (i >= n - degree - 1), 1.0, 0.0)
            return a1 + a2 + a3
        else:
            return a1

    clamped_nodes = t[..., i + k] == t[..., i]
    c1: Array = jnp.where(
        clamped_nodes,
        0.0,
        (x - t[..., i])
        / jnp.where(clamped_nodes, jnp.inf, (t[..., i + k] - t[..., i]))
        * _base_fun(x, k - 1, i, t, degree, extrapolate),
    )
    clamped_nodes = t[..., i + k + 1] == t[..., i + 1]
    c2: Array = jnp.where(
        t[..., i + k + 1] == t[..., i + 1],
        0.0,
        (t[..., i + k + 1] - x)
        / jnp.where(clamped_nodes, jnp.inf, (t[..., i + k + 1] - t[..., i + 1]))
        * _base_fun(x, k - 1, i + 1, t, degree, extrapolate),
    )

    return c1 + c2


@partial(jit, static_argnames=("degree", "open_spline"))
def eval_spline(x: Array, grid: Array, coefs: Array, degree: int = 3, open_spline: bool = False):
    if open_spline:
        n = grid.shape[-1] - degree - 1
    else:
        n = grid.shape[-1] - degree - 1 + 2 * degree
        
    assert coefs.shape[-1] == n, f"B-Spline has {n} basis functions! Only {coefs.shape[-1]} coefficients given."
    
    y = basis(x, grid, degree, open_spline=open_spline)
    return jnp.sum(y * coefs, axis=-1)


@partial(jit, static_argnames=("degree", "open_spline"))
def basis(x: Array, grid: Array, degree: int = 3, open_spline: bool = False) -> Array:
    """Computes the spine basis with respect to the provided grid.

    Parameters
    ----------
    x : Array
    grid : Array
        spline grid; if multidimensional, this will evaluate the spline for each dimension
        and the output can be seen as a compressed tensor.
    degree : int, optional
        spline degree

    Returns
    -------
    Array
    """
    
    if not open_spline:
        g0 = grid[..., 0:1]
        g1 = grid[..., -1:]
        grid = concatenate([jnp.repeat(g0, degree, axis=-1), grid, jnp.repeat(g1, degree, axis=-1)], axis=-1)
        extrapolate = True
    else:
        extrapolate = False
        
    n = grid.shape[-1] - degree - 1
    return vmap(lambda i: _base_fun(x, degree, i, grid, degree, extrapolate), 0, -1)(jnp.arange(n))
