from dataclasses import dataclass, field

from .prelude import *


class Domain(T.Protocol):
    dimension: int

    def support(self) -> ndarray:
        ...

    def includes(self, sample: ndarray) -> ndarray:  # should return boolean array
        ...
    
    def transform(self, uniform_sample: ndarray) -> ndarray:
        ...
    

@dataclass(slots=True, frozen=True)
class Hypercube:
    dimension: int = field(init=False, hash=False, repr=False)
    lb: tuple[float, ...]
    ub: tuple[float, ...]

    def __post_init__(self):
        assert len(self.lb) == len(self.ub)
        object.__setattr__(self, 'dimension', len(self.lb))

    def support(self) -> ndarray:
        lb, ub = array((self.lb, self.ub))
        return prod(ub - lb)

    def includes(self, sample: ndarray) -> ndarray:
        lb, ub = array((self.lb, self.ub))
        return (lb <= sample) & (sample <= ub)

    def transform(self, uniform_sample: ndarray) -> ndarray:
        lb, ub = array((self.lb, self.ub))
        return transform_hypercube(uniform_sample, lb, ub)

    def transform_bnd(self, uniform_sample: ndarray) -> ndarray:
        lb, ub = array((self.lb, self.ub))
        return transform_hypercube_bnd(uniform_sample, lb, ub)

    def normal_vec(self, bnd_sample: ndarray, rtol=1e-05, atol=1e-08) -> Optional[ndarray]:
        lb, ub = array((self.lb, self.ub))

        in_domain = (
            (bnd_sample > lb - (atol + rtol * abs(lb)))
            & (bnd_sample < ub + (atol + rtol * abs(ub)))
        ).all(axis=-1)
        if not in_domain.any():
            return None
        on_lb = jnp.isclose(bnd_sample, lb, rtol=rtol, atol=atol).astype(bnd_sample.dtype)
        on_ub = jnp.isclose(bnd_sample, ub, rtol=rtol, atol=atol).astype(bnd_sample.dtype)
        on_lb = where(jnp.cumsum(on_lb, axis = -1) > 1, 0., on_lb)
        on_ub = where(jnp.cumsum(on_ub, axis = -1) > 1, 0., on_ub)
        n = on_ub - on_lb
        
        if (norm(n, axis=-1) == 1.).all():
            return n
        else:
            return None


@dataclass(slots=True, frozen=True)
class Sphere:
    dimenson: int = field(default=3, init=False, hash=False, repr=False)
    radius: float
    origin: tuple[float, float, float]

    def __post_init__(self):
        assert len(array(self.radius).shape) == 0, "Radius must be a scalar."
        assert len(array(self.origin).shape) == 1, "Origin must be `tuple[float]`."

    def support(self):
        return 4 / 3 * array(self.radius) ** 3 * pi

    def includes(self, sample: ndarray) -> bool:
        r, o = array(self.radius), array(self.origin)
        return norm(sample - o, axis=-1) < r
    
    def transform(self, uniform_sample: ndarray) -> ndarray:
        r, o = array(self.radius), array(self.origin)
        return transform_sphere(uniform_sample, r, o)

    def transform_bnd(self, uniform_sample: ndarray) -> ndarray:
        r, o = array(self.radius), array(self.origin)
        return transform_sphere_bnd(uniform_sample, r, o)

    def normal_vec(self, bnd_sample: ndarray, rtol=1e-05, atol=1e-08) -> Optional[ndarray]:
        r, o = array(self.radius), array(self.origin)
        sample = bnd_sample - o

        sample_norm = norm(sample, axis=-1)
        on_bnd = jnp.isclose(sample_norm, r, rtol=rtol, atol=atol)
        if not on_bnd.all():
            return None
        return sample / sample_norm


@dataclass(slots=True, frozen=True)
class Circle(Sphere):
    dimenson: int = field(default=2, init=False, hash=False, repr=False)
    radius: float
    origin: tuple[float, float]

    def __post_init__(self):
        assert len(array(self.radius).shape) == 0, "Radius must be a scalar."
        assert len(array(self.origin).shape) == 1, "Origin must be `tuple[float]`."

    def support(self):
        return array(self.radius) ** 2 * pi
    
    def transform(self, uniform_sample: ndarray) -> ndarray:
        r, o = array(self.radius), array(self.origin)
        return transform_circle(uniform_sample, r, o)

    def transform_bnd(self, uniform_sample: ndarray) -> ndarray:
        r, o = array(self.radius), array(self.origin)
        return transform_circle_bnd(uniform_sample, r, o)

@dataclass(slots=True, frozen=True)
class Annulus:
    dimension: int = field(default=2, init=False, hash=False, repr=False)
    r1: float  # inner radius
    r2: float  # outer radius
    origin: tuple[float, float]

    def __post_init__(self):
        assert len(array(self.r1).shape) == 0, "Radius must be a scalar."
        assert len(array(self.r2).shape) == 0, "Radius must be a scalar."
        assert len(array(self.origin).shape) == 1, "Origin must be `tuple[float]`."

    def support(self):
        return (array(self.r2) ** 2 - array(self.r1) ** 2) * pi
    
    def transform(self, uniform_sample: ndarray) -> ndarray:
        r1, r2, o = array(self.r1), array(self.r2), array(self.origin)
        samples = transform_annulus(uniform_sample, r1, r2, o)
        return samples

    # def transform_bnd(self, uniform_sample: ndarray) -> ndarray:
    #     r, o = array(self.radius), array(self.origin)
    #     return transform_sphere_bnd(uniform_sample, r, o)

    # def normal_vec(self, bnd_sample: ndarray, rtol=1e-05, atol=1e-08) -> Optional[ndarray]:
    #     r, o = array(self.radius), array(self.origin)
    #     sample = bnd_sample - o

    #     sample_norm = norm(sample, axis=-1)
    #     on_bnd = jnp.isclose(sample_norm, r, rtol=rtol, atol=atol)
    #     if not on_bnd.all():
    #         return None
    #     return sample / sample_norm


def transform_hypercube(x: ndarray, lb: ndarray, ub: ndarray) -> ndarray:
    x = where(x > 1., x % 1., x)
    return x * (ub - lb) + lb


def transform_hypercube_bnd(x: ndarray, lb: ndarray, ub: ndarray) -> ndarray:
    assert len(lb.shape) == 1 and lb.shape[0] == ub.shape[0]
    dim = lb.shape[0]
    x = x % 1.
    if dim == 1:
        x = (x > 0.5).astype(u32)
        return transform_hypercube(x, lb, ub)
    if len(x.shape) == 0:
        x = x.ravel()
    if len(x.shape) == 1:
        x = x[:, None]
    assert x.shape[-1] == dim - 1
    idx = jnp.floor(2 * dim * x[..., 0]).astype(u32)
    x = x.at[..., 0].set(2 * dim * x[..., 0] - idx)
    bounds = (idx >= dim).astype(x.dtype)
    _insert = lambda x, i, v: jnp.insert(x, i, v, axis=-1)
    for axis in range(len(x.shape[:-1])):
        _insert = vmap(_insert, axis, axis)
    x = _insert(x, idx % dim, bounds)
    return transform_hypercube(x, lb, ub)


def transform_circle(x: ndarray, r: ndarray, o: ndarray) -> ndarray:
    # assert x.shape[-1] == 2
    # assert len(r.shape) == 0
    # assert o.shape[-1] == 2
    # x = x % 1.
    # x = (x - 0.5) * 2
    # x1 = x[..., 0] * sqrt(1-0.5*x[..., 1]**2)
    # x2 = x[..., 1] * sqrt(1-0.5*x[..., 0]**2)
    # return stack((x1, x2), -1) * r + o
    return transform_annulus(x, array(0.), r, o)


def transform_annulus(x: ndarray, r1: ndarray, r2: ndarray, o: ndarray) -> ndarray:
    assert x.shape[-1] == 2
    assert len(r1.shape) == 0 and len(r2.shape) == 0
    assert o.shape[-1] == 2
    x = x % 1.
    theta = 2 * pi * x[..., 0]
    r = sqrt(x[..., 1] * (r2 ** 2 - r1 ** 2) + r1 ** 2)
    x1 = r * cos(theta)
    x2 = r * sin(theta)
    return stack((x1, x2), -1) + o


def transform_sphere(x: ndarray, r: ndarray, o: ndarray) -> ndarray:
    assert x.shape[-1] == 3
    assert len(r.shape) == 0
    assert o.shape[-1] == 3
    x = x % 1.
    x1, x2, x3 = x[..., 0], x[..., 1], x[..., 2]
    _x1 = x1 * sqrt(1 - 0.5*x2**2 - 0.5*x3**2 + x2**2*x3**2 / 3)
    _x2 = x2 * sqrt(1 - 0.5*x3**2 - 0.5*x1**2 + x3**2*x1**2 / 3)
    _x3 = x3 * sqrt(1 - 0.5*x1**2 - 0.5*x2**2 + x1**2*x2**2 / 3)
    return stack((_x1, _x2, _x3), -1) * r + o


def transform_sphere_bnd(x: ndarray, r: ndarray, o: ndarray) -> ndarray:
    assert x.shape[-1] == 2
    assert len(r.shape) == 0
    assert o.shape[-1] == 3
    x = x % 1.
    angle = 2 * pi * x
    theta, phi = angle[..., 0], angle[..., 1]
    x1 = sin(theta) * cos(phi)
    x2 = sin(theta) * sin(phi)
    x3 = cos(theta)
    return stack((x1, x2, x3), -1) * r + o


def transform_circle_bnd(s: ndarray, r: ndarray, o: ndarray) -> ndarray:
    assert len(r.shape) == 0
    assert len(o.shape) == 1 and o.shape[0] == 2
    s = s % 1.
    phi = 4 * pi * s.squeeze()
    x1 = cos(phi)
    x2 = sin(phi)
    return stack((x1, x2), -1) * r + o


@partial(jit, static_argnames=('a', 'b', 'c'))
def transform_triangle(
    x: ndarray, 
    a: tuple[float, ...], b: tuple[float, ...], c: tuple[float, ...]
) -> ndarray:
    assert x.shape[-1] == 2
    i, j, k = array(a), array(b), array(c)
    assert i.shape == j.shape == k.shape
    x = x % 1.
    x = where(norm(x, 1, -1, keepdims=True) > 1., 1. - x, x)[..., None]
    d1 = (j - i)
    d2 = (k - i)
    return i + x[..., 0] * d1 + x[..., 1] * d2


@partial(jit, static_argnames='points')
def transform_polyline(
    s: ndarray, points: tuple[tuple[float, float], ...]
) -> ndarray:
    line = array(points)
    assert len(s.shape) == 0 or len(s.shape) == 1
    segments = norm(line[1:] - line[:-1], axis=1)
    length = jnp.sum(segments)
    cumlengths = jnp.cumsum(segments)
    s = s % 1
    s = (s * length).ravel()
    idx = jnp.argmax(s[:, None] <= cumlengths, axis=-1)
    
    p1, p2 = line[idx], line[idx + 1]
    d = (cumlengths[idx] - s)
    return p2 + (p1 - p2) * d[:, None] / (norm(p1 - p2, axis=-1, keepdims=True))

