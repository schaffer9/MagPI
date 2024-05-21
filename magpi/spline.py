from magpi.prelude import *


def bspline(x, grid, coefs, degree=3):
    b = bbasis(x, grid, degree)
    return b @ coefs


# @partial(jit, static_argnames=("degree",))
# def bbasis(x, grid, degree = 3):
#     # todo: optimize loop, arbitrary input shape, distance precomputing
#     m = len(grid)
#     n = m + degree - 1
#     N = zeros((n,))
#     k = jnp.searchsorted(grid, x) + degree - 1
#     grid = jnp.concatenate([jnp.repeat(grid[0], degree), grid, jnp.repeat(grid[-1], degree)])

#     # extrapolation:
#     k = lax.cond(x <= grid[0], lambda: k + 1, lambda: k)
#     k = lax.cond(x > grid[-1], lambda: k - 1, lambda: k)

#     N = N.at[k].set(1.0)
#     for d in range(1, degree + 1):
#         N = N.at[k - d].set((grid[k + 1] - x) / (grid[k + 1] - grid[k - d + 1]) * N[k - d + 1])
#         for z in range(0, d-1):
#             i = k - d + z + 1
#             N = N.at[i].set((x - grid[i]) / (grid[i + d] - grid[i]) * N[i]
#                             + (grid[i + d + 1] - x) / (grid[i + d + 1] - grid[i + 1]) * N[i + 1])
#         N = N.at[k].set((x - grid[k]) / (grid[k + d] - grid[k]) * N[k])

#     return N


def _base_fun(x, k, i, t, degree):
    # this is faster but takes longer to compile
    if k == 0:
        n = len(t) - k - 1
        a1 = jnp.where((t[i] <= x) & (x < t[i + 1]), 1.0, 0.0)
        # a2 and a3 for extrapolation
        a2 = jnp.where((x < t[0]) & (i <= degree), 1.0, 0.0)
        a3 = jnp.where((x >= t[-1]) & (i >= n - degree - 1), 1.0, 0.0)
        return a1 + a2 + a3

    c1 = jnp.where(
        t[i + k] == t[i],
        0.0,
        ((x - t[i]) / (t[i + k] - t[i])
         * _base_fun(x, k - 1, i, t, degree)),
    )
    c2 = jnp.where(
        t[i + k + 1] == t[i + 1],
        0.0,
        ((t[i + k + 1] - x) / (t[i + k + 1] - t[i + 1])
         * _base_fun(x, k - 1, i + 1, t, degree)),
    )
    return c1 + c2


@partial(jit, static_argnames=("degree"))
def bbasis(x, t, degree=3):
    n = len(t) + degree - 1
    t = jnp.concatenate([jnp.repeat(t[0], degree), t, jnp.repeat(t[-1], degree)])
    return vmap(_base_fun, (None, None, 0, None, None), -1)(
        x, degree, jnp.arange(n), t, degree
    )


def fit_bspline(x, y, grid, degree=3):
    # C = vmap(bbasis, (0, None, None))(x, grid, degree)
    C = bbasis(x, grid, degree)
    Cinv = jnp.linalg.pinv(C)
    return Cinv @ y


def adjust_grid(x, grid, coefs, grid_eps=0.1, degree=3):
    # y = vmap(bspline, (0, None, None, None))(x, grid, coefs, degree)
    y = bspline(x, grid, coefs, degree)

    x_min = jnp.min(x)
    x_max = jnp.max(x)
    h = len(grid)
    grid_uniform = jnp.linspace(x_min, x_max, h)
    grid_adaptive = jnp.quantile(x, jnp.linspace(0, 1, len(grid)))
    grid = grid_eps * grid_uniform + (1 - grid_eps) * grid_adaptive
    c_new = fit_bspline(x, y, grid, degree)
    return grid, c_new
