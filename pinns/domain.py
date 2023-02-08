"""
This module contains some useful transformations. All transformations 
perform a uniform mapping from :math:`[0,1]^d` to the respective domain.
"""

from dataclasses import MISSING

from flax.struct import dataclass, field
from itertools import repeat, chain
from jax.scipy.stats.norm import ppf
from jaxopt.linear_solve import solve_lu

from .prelude import *

Array = ndarray
BoolArray = Array


class Domain(T.Protocol):
    dimension: property

    def support(self) -> Array:
        ...

    def transform(self, uniform_sample: Array) -> Array:
        ...


@dataclass
class Hypercube:
    lb: tuple[float, ...]
    ub: tuple[float, ...]
    dimension = property(lambda self: len(self.lb))

    def __post_init__(self):
        assert len(self.lb) == len(self.ub)

    def support(self) -> Array:
        lb, ub = array((self.lb, self.ub))
        return prod(ub - lb)

    def includes(self, sample: Array) -> Array:
        lb, ub = array((self.lb, self.ub))
        return (lb <= sample) & (sample <= ub)

    def transform(self, uniform_sample: Array) -> Array:
        lb, ub = array((self.lb, self.ub))
        return transform_hypercube(uniform_sample, lb, ub)

    def transform_bnd(self, uniform_sample: Array) -> Array:
        lb, ub = array((self.lb, self.ub))
        return transform_hypercube_bnd(uniform_sample, lb, ub)

    def normal_vec(self, bnd_sample: Array, rtol=1e-05, atol=1e-08) -> Optional[Array]:
        lb, ub = array((self.lb, self.ub))

        in_domain = (
            (bnd_sample > lb - (atol + rtol * abs(lb)))
            & (bnd_sample < ub + (atol + rtol * abs(ub)))
        ).all(axis=-1)
        if not in_domain.any():
            return None
        on_lb = jnp.isclose(bnd_sample, lb, rtol=rtol, atol=atol).astype(
            bnd_sample.dtype
        )
        on_ub = jnp.isclose(bnd_sample, ub, rtol=rtol, atol=atol).astype(
            bnd_sample.dtype
        )
        on_lb = where(jnp.cumsum(on_lb, axis=-1) > 1, 0.0, on_lb)
        on_ub = where(jnp.cumsum(on_ub, axis=-1) > 1, 0.0, on_ub)
        n = on_ub - on_lb

        if (norm(n, axis=-1) == 1.0).all():
            return n
        else:
            return None


@dataclass
class Parallelogram:
    a: tuple[float, ...]
    b: tuple[float, ...]
    c: tuple[float, ...]
    B: Array = field(repr=False, init=False)
    dimension = property(lambda self: len(self.a))

    def __post_init__(self):
        assert len(self.a) == len(self.b) == len(self.c)
        B, _ = affine_plane(*array([self.a, self.b, self.c]))
        object.__setattr__(self, "B", B)

    def support(self) -> Array:
        a, b, c = array(self.a), array(self.b), array(self.c)
        d = stack([b - a, c - a])
        return sqrt(jnp.linalg.det(d @ d.T))

    def transform(self, samples: Array) -> Array:
        assert samples.shape[-1] == 2
        samples = where(samples > 1, samples % 1., samples)
        a = array(self.a)
        if len(samples.shape) == 1:
            return self.B @ samples + a
        else:
            return samples @ self.B.T + a

    def transform_bnd(self, samples: Array) -> Array:
        a, b, c = array([self.a, self.b, self.c])
        return transform_polyline(
            samples,
            (a, a + b, a + b + c, c, a)
        )


def linear_map(X_ref: Array, X: Array) -> Array:
    assert X_ref.shape[0] == X.shape[0]
    B = solve_lu(lambda x: X_ref @ x, X)
    return B.T


def affine_plane(a: Array, b: Array, c: Array) -> tuple[Array, Array, Array]:
    X_ref = array([[0, 0, 1], [1, 0, 1], [0, 1, 1.0]])
    X = stack([a, b, c])
    B = linear_map(X_ref, X)
    return B[..., :-1], B[..., -1]


class _Spherical:
    radius: float
    origin: tuple[float, ...]

    def includes(self, sample: Array) -> bool:
        r, o = array(self.radius), array(self.origin)
        return norm(sample - o, axis=-1) < r

    def normal_vec(self, bnd_sample: Array, rtol=1e-05, atol=1e-08) -> Optional[Array]:
        r, o = array(self.radius), array(self.origin)
        sample = bnd_sample - o

        sample_norm = norm(sample, axis=-1)
        on_bnd = jnp.isclose(sample_norm, r, rtol=rtol, atol=atol)
        if not on_bnd.all():
            return None
        return sample / sample_norm


@dataclass
class Sphere(_Spherical):
    radius: float
    origin: tuple[float, float, float]
    dimension = property(lambda self: 3)

    def __post_init__(self):
        assert len(array(self.radius).shape) == 0, "Radius must be a scalar."
        assert len(array(self.origin).shape) == 1, "Origin must be `tuple[float]`."

    def support(self):
        return 4 / 3 * array(self.radius) ** 3 * pi

    def transform(self, uniform_sample: Array) -> Array:
        r, o = array(self.radius), array(self.origin)
        return transform_sphere(uniform_sample, r, o)

    def transform_bnd(self, uniform_sample: Array) -> Array:
        r, o = array(self.radius), array(self.origin)
        return transform_sphere_bnd(uniform_sample, r, o)


@dataclass
class Disk(_Spherical):
    radius: float
    origin: tuple[float, float]
    dimension = property(lambda self: 2)

    def __post_init__(self):
        assert len(array(self.radius).shape) == 0, "Radius must be a scalar."
        assert len(array(self.origin).shape) == 1, "Origin must be `tuple[float]`."

    def support(self):
        return array(self.radius) ** 2 * pi

    def transform(self, uniform_sample: Array) -> Array:
        r, o = array(self.radius), array(self.origin)
        return transform_circle(uniform_sample, r, o)

    def transform_bnd(self, uniform_sample: Array) -> Array:
        r, o = array(self.radius), array(self.origin)
        return transform_circle_bnd(uniform_sample, r, o)


@dataclass
class Annulus:
    r1: float  # inner radius
    r2: float  # outer radius
    origin: tuple[float, float]
    dimension = property(lambda self: 2)

    def __post_init__(self):
        assert len(array(self.r1).shape) == 0, "Radius must be a scalar."
        assert len(array(self.r2).shape) == 0, "Radius must be a scalar."
        assert len(array(self.origin).shape) == 1, "Origin must be `tuple[float]`."

    def support(self):
        return (array(self.r2) ** 2 - array(self.r1) ** 2) * pi

    def transform(self, uniform_sample: Array) -> Array:
        r1, r2, o = array(self.r1), array(self.r2), array(self.origin)
        samples = transform_annulus(uniform_sample, r1, r2, o)
        return samples


def transform_hypercube(x: Array, lb: Array, ub: Array) -> Array:
    x = where(x > 1.0, x % 1.0, x)
    return x * (ub - lb) + lb


def transform_hypercube_bnd(x: Array, lb: Array, ub: Array) -> Array:
    assert len(lb.shape) == 1 and lb.shape[0] == ub.shape[0]
    dim = lb.shape[0]
    x = x % 1.0
    if dim == 1:
        x = (x > 0.5).astype(u32)
        return transform_hypercube(x, lb, ub)
    scalar_input = len(x.shape) == 0
    if scalar_input:
        x = x.ravel()
    if len(x.shape) == 1 and dim == 2:
        x = x[:, None]

    msg = f"Input must be {dim - 1} dimensional!"
    assert x.shape[-1] == dim - 1, msg

    def insert_rules(lb, ub):
        b = jnp.abs(ub - lb)

        def support(i):
            return jnp.prod(concatenate([b[:i], b[i + 1 :]]))

        supports = [support(i) for i in range(len(b))]
        supports = array(supports + supports)
        cum_supports = jnp.cumsum(supports) / jnp.sum(supports)
        vals = list(chain(repeat(0.0, len(b)), repeat(1.0, len(b))))
        return supports / jnp.sum(supports), cum_supports, array(vals)

    supports, cum_supports, vals = insert_rules(lb, ub)

    def insert(x):
        i = jnp.argmax(x[0] < cum_supports)
        l = supports[i]
        x = lax.cond(
            i == 0,
            lambda: x.at[0].set(x[0] / l),
            lambda: x.at[0].set((x[0] - cum_supports[i - 1]) / l),
        )
        return jnp.insert(x, i % dim, vals[i])

    x = jnp.apply_along_axis(insert, -1, x)
    x = transform_hypercube(x, lb, ub)
    return x[0] if scalar_input else x


def transform_circle(x: Array, r: Array, o: Array) -> Array:
    return transform_annulus(x, array(0.0), r, o)


def transform_annulus(x: Array, r1: Array, r2: Array, o: Array) -> Array:
    assert x.shape[-1] == 2
    assert len(r1.shape) == 0 and len(r2.shape) == 0
    assert o.shape[-1] == 2
    x = x % 1.0
    theta = 2 * pi * x[..., 0]
    r = sqrt(x[..., 1] * (r2 ** 2 - r1 ** 2) + r1 ** 2)
    x1 = r * cos(theta)
    x2 = r * sin(theta)
    return stack((x1, x2), -1) + o


def transform_sphere(x: Array, r: Array, o: Array) -> Array:
    assert x.shape[-1] == 3
    assert len(r.shape) == 0
    assert o.shape[-1] == 3
    x = x % 1.0
    u = 2 * x[..., 0] - 1
    phi = 2 * pi * x[..., 1]
    rad = x[..., 2] ** (1 / 3)
    x1 = rad * cos(phi) * sqrt(1 - u ** 2)
    x2 = rad * sin(phi) * sqrt(1 - u ** 2)
    x3 = rad * u
    return stack((x1, x2, x3), -1) * r + o


def transform_sphere_bnd(x: Array, r: Array, o: Array) -> Array:
    assert x.shape[-1] == 2
    assert len(r.shape) == 0
    assert o.shape[-1] == 3
    x = x % 1.0
    x3, phi = 2 * x[..., 0] - 1, 2 * pi * x[..., 1]
    x1 = cos(phi) * (1 - x3 ** 2) ** 0.5
    x2 = sin(phi) * (1 - x3 ** 2) ** 0.5
    return stack((x1, x2, x3), -1) * r + o


def transform_circle_bnd(s: Array, r: Array, o: Array) -> Array:
    assert len(r.shape) == 0
    assert len(o.shape) == 1 and o.shape[0] == 2
    s = s % 1.0
    phi = 4 * pi * s.squeeze()
    x1 = cos(phi)
    x2 = sin(phi)
    return stack((x1, x2), -1) * r + o


def transform_triangle(
    x: Array, a: Array, b: Array, c: Array
) -> Array:
    assert x.shape[-1] == 2
    i, j, k = a, b, c
    assert i.shape == j.shape == k.shape
    x = x % 1.0
    x = where(norm(x, 1, -1, keepdims=True) > 1.0, 1.0 - x, x)[..., None]
    d1 = j - i
    d2 = k - i
    return i + x[..., 0] * d1 + x[..., 1] * d2


def transform_polyline(s: Array, points: tuple[Array]) -> Array:
    line = stack(points)
    assert len(s.shape) == 0 or len(s.shape) == 1
    segments = norm(line[1:] - line[:-1], axis=1)
    length = jnp.sum(segments)
    cumlengths = jnp.cumsum(segments)
    s = s % 1
    s = (s * length).ravel()
    idx = jnp.argmax(s[:, None] <= cumlengths, axis=-1)

    p1, p2 = line[idx], line[idx + 1]
    d = cumlengths[idx] - s
    return p2 + (p1 - p2) * d[:, None] / (norm(p1 - p2, axis=-1, keepdims=True))
