from typing import Callable

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

    c1: Array = jnp.where(
        t[..., i + k] == t[..., i],
        0.0,
        (x - t[..., i]) / (t[..., i + k] - t[..., i]) * _base_fun(x, k - 1, i, t, degree, extrapolate)
    )
    c2: Array = jnp.where(
        t[..., i + k + 1] == t[..., i + 1],
        0.0,
        (t[..., i + k + 1] - x) / (t[..., i + k + 1] - t[..., i + 1]) * _base_fun(x, k - 1, i + 1, t, degree, extrapolate),
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
        input
    grid : Array
        spline grid
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


def make_grid(start: float, stop: float, power: float, num: int) -> Array:
    """Crates a grid with power spacing centered at the center of
    `start` and `stop`.

    Parameters
    ----------
    start : float
    stop : float
    power : float
    num : int

    Returns
    -------
    Array
    """
    t = jnp.linspace(-1, 1, num)
    t = jnp.sign(t) * (jnp.abs(t) ** power)
    t = (t + 1) / 2
    t = t * jnp.abs(start - stop) + start
    return t


def _grid_init(node_min, node_max, grid_power, nodes):
    def init(shape, dtype):
        t = make_grid(node_min, node_max, grid_power, nodes)
        return nn.initializers.constant(t)(random.key(0), shape, dtype)

    return init


class SplineActivation(nn.Module):
    """A Spline activation function module.

    This module adds a trainable spline function to a standard activation
    function.

    Parameters
    ----------
    nodes : int
        number of grid nodes, default is -3.0
        There are `nodes - degree - 1` coefficients for each activation
    node_min : float
        placement of the starting node of the grid, default is -3.0
    node_max : float
        placement of the final node of the grid, default is 3.0
    degree : int
        spline degree, default is 3
    grid_power : float
        the power spacing of the grid, default is 2.0
    activation : Callable
        base activation function, default is `tanh`
    coef_init : Initializer
        initializer for the spline coefficients, default is zero initialization
    parameterize_grid : bool
        if `True`, the spline grid is found inside the `grids` collection and
        can be changed or trained, this allows for a different grid for each activation,
        default is `False`

    """

    nodes: int = 14
    node_min: float = -3.0
    node_max: float = 3.0
    degree: int = 3
    grid_power: float = 2.0
    activation: Callable = nn.tanh
    coef_init: nn.initializers.Initializer = nn.initializers.zeros_init()
    parameterize_grid: bool = False

    @nn.compact
    def __call__(self, x) -> Array:
        if x.shape == ():
            x = x.ravel()

        if self.parameterize_grid:
            grid = self.variable(
                "grids",
                "grid",
                _grid_init(self.node_min, self.node_max, self.grid_power, self.nodes),
                (x.shape[-1], self.nodes),
                x.dtype,
            )
            grid = grid.value
        else:
            grid = make_grid(self.node_min, self.node_max, self.grid_power, self.nodes)

        coefs = self.param("coefs", self.coef_init, (x.shape[-1], self.nodes - self.degree - 1))

        y = jnp.sum((basis(x, grid, degree=self.degree, open_spline=True) * coefs), axis=-1)
        return self.activation(x) + y
