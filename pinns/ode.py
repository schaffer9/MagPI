"""
This code was adoped from https://implicit-layers-tutorial.org/implicit_functions/
"""

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


def odeint(df, y, ts, *args, unroll=1, method=rk4):
    return _odeint(df, unroll, method, y, ts, *args)


#@partial(jax.custom_jvp, nondiff_argnums=(0, 1, 2))
def _odeint(df, unroll, method, y, ts, *args):
    @partial(jax.checkpoint, policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
    def _step(state, t):
        y, t_prev = state
        dt = t - t_prev
        y = method(df, y, t, dt, *args)
        return (y, t), y

    _, ys = lax.scan(_step, (y, ts[0]), ts[1:], unroll=unroll)
    return tree_map(lambda a, b: concatenate([a[None], b]), y, ys)


# @_odeint.defjvp
# def _odeint_jvp(df, unroll, method, primals, tangents):
#     print("fooo")
#     y0, ts, *args = primals
#     delta_y0, _, *delta_args = tangents
    
#     def df_aug(aug_state, t, args, delta_args):
#         primal_state, tangent_state = aug_state
#         primal_dot, tangent_dot = jax.jvp(df, (primal_state, t, *args), (tangent_state, 0., *delta_args))
#         return (primal_dot, tangent_dot)

#     aug_init_state = (y0, delta_y0)
#     ys, ys_dot = odeint(df_aug, aug_init_state, ts, args, delta_args, method=method, unroll=unroll)
#     return ys, ys_dot
