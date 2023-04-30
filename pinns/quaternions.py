from .prelude import *
from dataclasses import dataclass
from jax.tree_util import register_pytree_node_class

Scalar = Array | float | int
Vec = Array | tuple[Scalar, Scalar, Scalar]


@register_pytree_node_class
@dataclass(frozen=True, slots=True, weakref_slot=True)
class Quaternion:
    _a: Scalar
    _q: Vec
    real = property(lambda self: self._a)
    imag = property(lambda self: self._q)

    def __post_init__(self):
        if not isinstance(self._a, Array):
            object.__setattr__(self, "_a", array(self._a))
            
        if not isinstance(self._q, Array):
            object.__setattr__(self, "_q", array(self._q))
        
        assert self._a.shape == ()
        assert self._q.shape == (3,)
        assert self._a.dtype == self._q.dtype

    @classmethod
    def zero_quaternion(cls):
        return cls(0., (0., 0., 0.))

    def __eq__(self, other: 'Quaternion'):
        assert isinstance(other, self.__class__)
        real_eq = self.real == other.real
        imag_eq = jnp.all(self.imag == other.imag)
        return real_eq and imag_eq

    def __neg__(self) -> 'Quaternion':
        return self.__class__(-self.a, -self.q)

    def __add__(self, other: 'Quaternion') -> 'Quaternion':
        if isinstance(other, self.__class__):
            a = self.real + other.real
            q = self.imag + other.imag
        else:
            a = self.real + other
            q = self.imag + other
        return self.__class__(a, q)
    
    def __radd__(self, other: 'Quaternion') -> 'Quaternion':
        return self + other
    
    def __sub__(self, other: 'Quaternion') -> 'Quaternion':
        if isinstance(other, self.__class__):
            a = self.real - other.real
            q = self.imag - other.imag
        else:
            a = self.real - other
            q = self.imag - other
        return self.__class__(a, q)
    
    def __rsub__(self, other: 'Quaternion') -> 'Quaternion':
        return self - other

    def __mul__(self, other: Union['Quaternion', Scalar]) -> 'Quaternion':
        if isinstance(other, self.__class__):
            a = self.real * other.real - self.imag @ other.imag
            q = self.real * other.imag + other.real * self.imag + cross(self.imag, other.imag)
        else:
            a = self.real * other
            q = self.imag * other
        return self.__class__(a, q)
    
    def __rmul__(self, other: 'Quaternion') -> 'Quaternion':
        return self * other

    def __abs__(self) -> Scalar:
        return sqrt(self.real ** 2 + jnp.sum(self.imag * self.imag))
    
    def __pow__(self, pow: Scalar) -> 'Quaternion':
        l = pow * quanternion_log(self)
        return quanternion_exp(l)
    
    def __rpow__(self, base: Scalar) -> 'Quaternion':
        b = jnp.log(base)
        return quanternion_exp(b * self)
        
    def __truediv__(self, other: Scalar) -> 'Quaternion':
        return self * (1 / other)
    
    # def __rtruediv__(self, other: Scalar) -> 'Quaternion':
    #     return self * (1 / other)

    def reciprocal(self) -> 'Quaternion':
        return self.conj() * (1 / abs(self) ** 2)
    
    def conj(self) -> 'Quaternion':
        return self.__class__(self.real, -self.imag)
    
    def exp(self, p) -> 'Quaternion':
        z = exp(self.real)
        r = norm(self.imag)
        a = z * cos(r)
        q = z * sgn(self.imag) * sin(r)
        return self.__class__(a, q)

    def tree_flatten(self):
        children = (self._a, self._q)  # arrays / dynamic values
        aux_data = None  # static values
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children[0], children[1])
    

def quanternion_exp(q: Quaternion) -> Quaternion:
    z = exp(q.real)
    r = norm(q.imag)
    a = z * cos(r)
    q = z * sgn(q.imag) * sin(r)
    print('exp', a.shape, q.shape)
    return Quaternion(a, q)

def quanternion_log(q: Quaternion) -> Quaternion:
    r = abs(q)
    a = jnp.log(r)
    q = sgn(q.imag) * arg(q)
    print('log', a.shape, q.shape)
    return Quaternion(a, q)


def arg(q: Quaternion) -> Scalar:
    return jnp.arccos(q.real * (1 / abs(q)))


def sgn(v: Array) -> Array:
    eps = jnp.finfo(v.dtype).eps
    r = norm(v)
    return lax.cond(
        r <= eps,
        lambda: jnp.zeros_like(v),
        lambda: v / r,
    )


def quaternion_rotation(x: Vec, q: Quaternion) -> Vec:
    qinv = q.reciprocal()
    p = Quaternion(zeros((), dtype=x.dtype), x)
    return (qinv * p * q).imag


def quaternion_reflection(x: Vec, normal_vec: Vec) -> Vec:
    n = array(normal_vec)
    n = n / norm(n)
    z = zeros((), dtype=x.dtype)
    x = Quaternion(z, x)
    q = Quaternion(z, normal_vec)
    return (q * x * q).imag


def from_euler_angles(angles: Vec) -> Quaternion:
    angles = array(angles)
    angles = angles / 2
    s = sin(angles)
    c = cos(angles)
    a = prod(c) + prod(s)
    q = (
        s[0] * c[1] * c[2] - c[0] * s[1] * s[2],
        c[0] * s[1] * c[2] + s[0] * c[1] * s[2],
        c[0] * c[1] * s[2] - s[0] * s[1] * c[2],
    )
    return Quaternion(a, q)


def from_axis_angle(angle: Scalar, axis: Vec) -> Quaternion:
    axis = array(axis)
    angle = angle / 2
    return Quaternion(cos(angle), axis * sin(angle))
    