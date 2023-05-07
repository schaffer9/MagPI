from .prelude import *

from jaxopt.linear_solve import solve_lu
from chex import Array

Samples = Array
PartitionPart = list[Samples]
Interpolation = Callable[[Array], Array]


def max_dist(x0: Samples, x1: Samples) -> float:
    def _max_dist(x, y):
        d = norm(x - y, axis=-1)
        return jnp.sort(d)[-1]

    d = vmap(_max_dist, (0, None))(x0, x1)
    return d.max()


def scatter_interpolate(
    x0: Samples, x1: Samples, r: float | None = None
) -> Interpolation:
    if r is None:
        r = max_dist(x0, x1) / 2
    mk = len(x0) + len(x1)
    X: Array = concatenate([x0, x1])
    e = 1.25 * r / sqrt(mk)
    phi = lambda x, y: 1 / sqrt(e**2 + norm(x - y, axis=-1) ** 2)
    a0 = vmap(phi, (0, None))(x0, X)
    a1 = vmap(phi, (0, None))(x1, X)
    a = concatenate([a0, a1])
    s = X.shape
    A = jnp.block(
        [
            [a, X, ones((s[0], 1))],
            [X.T, zeros((s[1], s[1])), zeros((s[1], 1))],
            [ones((1, s[0])), zeros((1, s[1])), zeros((1, 1))],
        ]
    )
    b: Array = concatenate([zeros(len(x0)), ones(len(x1)), zeros(s[1]), zeros(1)])
    params = solve_lu(lambda x: A @ x, b)
    p1, p2, p3 = params[: s[0]], params[s[0] : s[0] + s[1]], params[-1]
    # return lambda x: concatenate([phi(x, X), x, ones(1)]) @ params
    return lambda x: phi(x, X) @ p1 + x @ p2 + p3


def shape_function(
    p1: PartitionPart, p2: PartitionPart, mu: int | None = None, r: float | None = None
) -> Interpolation:
    """Creates a indicator function which is zero on the boundary as in [1]_.
    `p1` and `p2` should cover the respective boundary domain.
    A spline function is trained for each subset in `p1` to the
    respective subset in `p2` and vice versa. The samples from each
    subset should not overlap and the opposing set should not be neighboring
    each other.

    Parameters
    ----------
    p1 : Partition
    p2 : Partition
    mu : int | None, optional
        the exponent to compute the final indicator function, by
        default it is the number of subsets on the boundary domain
    r : float | None, optional
        the shape parameter of the spline function should be the minimum radius of the circle
        which includes all sample points, if not given it is computed for each subset.

    Returns
    -------
    Interpolation

    Notes
    -----
    .. [1] Sheng, Hailong, and Chao Yang. "PFNN: A penalty-free neural network
        method for solving a class of second-order boundary-value problems on
        complex geometries." Journal of Computational Physics 428 (2021): 110085.

    """
    if mu is None:
        mu = len(p1) + len(p2)
    lks = [scatter_interpolate(_x0, _x1, r) for _x0, _x1 in zip(p1, p2)] + [
        scatter_interpolate(_x0, _x1, r) for _x0, _x1 in zip(p2, p1)
    ]

    def _l(x):
        yk = stack([lk(x) for lk in lks])
        yk = 1 - (1 - yk) ** mu
        return jnp.prod(yk)

    return _l


def octal_partitioning(x: Samples) -> tuple[PartitionPart, PartitionPart]:
    """If all samples are between -1 and 1, this function will create a
    partitioning of the domain where each octant is mapped to the
    opposing one.

    Parameters
    ----------
    x : Samples

    Returns
    -------
    tuple[PartitionPart, PartitionPart]
    """
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
