"""
This code was adoped from https://implicit-layers-tutorial.org/implicit_functions/
"""
from chex import ArrayTree

from .prelude import *


def rk4(df, y, t, dt, *args):
    y1 = tree_scalar_mul(dt, df(y, t, *args))
    y2 = tree_scalar_mul(dt, df(tree_add(y, tree_scalar_mul(1 / 2, y1)), t + dt / 2, *args))
    y3 = tree_scalar_mul(dt, df(tree_add(y, tree_scalar_mul(1 / 2, y2)), t + dt / 2, *args))
    y4 = tree_scalar_mul(dt, df(tree_add(y, y3), t + dt, *args))
    y = tree_map(lambda y, y1, y2, y3, y4: y + 1 / 6 * (y1 + 2 * y2 + 2 * y3 + y4), y, y1, y2, y3, y4)
    return y


def explicit_euler(df, y, t, dt, *args):
    dy = df(y, t, *args)
    y = tree_add(y, tree_scalar_mul(dt, dy))
    return y


def odeint(df: Callable, y: ArrayTree, ts: Array, *args, unroll: int=1, method: Callable=rk4) -> ArrayTree:
    """
    ODE integration.
    This function fully supports forward and backward automatic differentiation by using
    `jax.checkpoint`.

    Parameters
    ----------
    df : Callable
        function describing the dynamics of the system
    y : ArrayTree
        initial value
    ts : Array
        timesteps
    unroll : int, optional
        by default 1
    method : Callable, optional
        method for integration, by default rk4

    Returns
    -------
    ArrayTree
    """
    return _odeint(df, unroll, method, y, ts, *args)


def _odeint(df, unroll, method, y, ts, *args):
    @partial(jax.checkpoint, policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
    def _step(state, t):
        y, t_prev = state
        dt = t - t_prev
        y = method(df, y, t, dt, *args)
        return (y, t), y

    _, ys = lax.scan(_step, (y, ts[0]), ts[1:], unroll=unroll)
    return tree_map(lambda a, b: concatenate([a[None], b]), y, ys)
