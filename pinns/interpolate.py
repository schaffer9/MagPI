from .prelude import *

from jaxopt.linear_solve import solve_lu


Array = Any
Nodes = Array
Partition = list[Nodes]
Interpolation = Callable[[Array], Array]


def max_dist(x0: Nodes, x1: Nodes) -> float:
    def _max_dist(x, y):
        d = norm(x - y, axis=-1)
        return jnp.sort(d)[-1]
    d = vmap(_max_dist, (0, None))(x0, x1)
    return d.max()


def scatter_interpolate(
    x0: Nodes, 
    x1: Nodes, 
    r: float | None = None
) -> Interpolation:  
    if r is None:
        r = max_dist(x0, x1) / 2
    mk = len(x0) + len(x1)
    X: Array = concatenate([x0, x1])
    e = 1.25 * r / sqrt(mk)
    phi = lambda x, y: 1 / sqrt(e ** 2 + norm(x - y, axis=-1) ** 2)
    a0 = vmap(phi, (0, None))(x0, X)
    a1 = vmap(phi, (0, None))(x1, X)
    a = concatenate([a0, a1])
    s = X.shape
    A = jnp.block([
        [a, X, ones((s[0], 1))],
        [X.T, zeros((s[1], s[1])), zeros((s[1], 1))],
        [ones((1, s[0])), zeros((1, s[1])), zeros((1, 1))]
    ])
    b: Array = concatenate(
        [zeros(len(x0)), ones(len(x1)), zeros(s[1]), zeros(1)]
    )
    params = solve_lu(lambda x: A @ x, b)
    return lambda x: concatenate([phi(x, X), x, ones(1)]) @ params


def shape_function(
    p1: Partition, 
    p2: Partition,
    mu: int | None = None,
    r: float | None = None
) -> Interpolation:
    if mu is None:
        mu = len(p1) + len(p2)
    lks = (
        [scatter_interpolate(_x0, _x1, r) for _x0, _x1 in zip(p1, p2)] + 
        [scatter_interpolate(_x0, _x1, r) for _x0, _x1 in zip(p2, p1)]
    )

    def _l(x):
        yk = stack([lk(x) for lk in lks])
        yk = 1 - (1 - yk) ** mu
        return jnp.prod(yk)

    return _l


def octal_partition(x: Nodes) -> tuple[Partition, Partition]:
    assert x.shape[1] == 3, "Nodes must be three dimensional"
    p1 = [
        x[(x[:, 0] >= 0) & (x[:, 1] >= 0) & (x[:, 2] >= 0)],
        x[(x[:, 0] >= 0) & (x[:, 1] >= 0) & (x[:, 2] <= 0)],
        x[(x[:, 0] >= 0) & (x[:, 1] <= 0) & (x[:, 2] >= 0)],
        x[(x[:, 0] >= 0) & (x[:, 1] <= 0) & (x[:, 2] <= 0)],
    ]
    p2 = [
        x[(x[:, 0] < 0) & (x[:, 1] < 0) & (x[:, 2] < 0)],
        x[(x[:, 0] < 0) & (x[:, 1] < 0) & (x[:, 2] > 0)],
        x[(x[:, 0] < 0) & (x[:, 1] > 0) & (x[:, 2] < 0)],
        x[(x[:, 0] < 0) & (x[:, 1] > 0) & (x[:, 2] > 0)],
    ]
    return p1, p2
