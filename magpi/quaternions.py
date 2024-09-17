from dataclasses import dataclass
from typing import Self

from jax.tree_util import register_pytree_node_class

from .prelude import *

Scalar = Array | float | int
Vec = Array | tuple[Scalar, Scalar, Scalar]


@register_pytree_node_class
@dataclass(frozen=True, slots=True, weakref_slot=True)
class Quaternion:
    _a: Array
    _q: Array
    real = property(lambda self: self._a)
    imag = property(lambda self: self._q)

    def __init__(self, a: Scalar, q: Vec):
        object.__setattr__(self, "_a", asarray(a))
        object.__setattr__(self, "_q", asarray(q))
        assert self._a.shape == ()
        assert self._q.shape == (3,)
        assert self._a.dtype == self._q.dtype

    @classmethod
    def zero_quaternion(cls):
        return cls(0.0, (0.0, 0.0, 0.0))

    def __eq__(self, other: object):
        if not isinstance(other, self.__class__):
            return NotImplemented
        
        real_eq = self.real == other.real
        imag_eq = jnp.all(self.imag == other.imag)
        return real_eq and imag_eq

    def __neg__(self) -> Self:
        return self.__class__(-self._a, -self._q)

    def __add__(self, other: Self | Scalar) -> Self:
        if isinstance(other, self.__class__):
            a = self.real + other.real
            q = self.imag + other.imag
        else:
            a = self.real + other
            q = self.imag + other
        return self.__class__(a, q)

    def __radd__(self, other) -> Self:
        return self + other

    def __sub__(self, other: Self | Scalar) -> Self:
        if isinstance(other, self.__class__):
            a = self.real - other.real
            q = self.imag - other.imag
        else:
            a = self.real - other
            q = self.imag - other
        return self.__class__(a, q)

    def __rsub__(self, other) -> Self:
        return self - other

    def __mul__(self, other: Self | Scalar) -> Self:
        if isinstance(other, self.__class__):
            a = self.real * other.real - self.imag @ other.imag
            q = (
                self.real * other.imag
                + other.real * self.imag
                + cross(self.imag, other.imag)
            )
        else:
            a = self.real * other
            q = self.imag * other
        return self.__class__(a, q)

    def __rmul__(self, other) -> Self:
        return self * other

    def __abs__(self) -> Scalar:
        return sqrt(self.real**2 + jnp.sum(self.imag * self.imag))

    def __pow__(self, pow: Scalar) -> Self:
        e = quanternion_log(self) * pow
        return quanternion_exp(e)

    def __rpow__(self, base) -> Self:
        b = jnp.log(base)
        return quanternion_exp(self * b)
    
    def __truediv__(self, other: Scalar) -> Self:
        if isinstance(other, self.__class__):
            return quaternion_right_div(self, other)
        else:
            return self * (1 / other)

    def reciprocal(self) -> Self:
        return self.conj() * (1 / abs(self) ** 2)

    def conj(self) -> Self:
        return self.__class__(self.real, -self.imag)

    def tree_flatten(self):
        children = (self._a, self._q)  # arrays / dynamic values
        return (children, None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children[0], children[1])


def quaternion_right_div(q1: Quaternion, q2: Quaternion) -> Quaternion:
    return q1 * q2.reciprocal()


def quaternion_left_div(q1: Quaternion, q2: Quaternion) -> Quaternion:
    return q2.reciprocal() * q1
   

def quanternion_exp(q: Quaternion) -> Quaternion:
    """
    Computes :math:`\\exp(q)`

    Parameters
    ----------
    q : Quaternion

    Returns
    -------
    Quaternion
    """
    z = exp(q.real)
    r = norm(q.imag)
    a = z * cos(r)
    p = z * sgn(q.imag) * sin(r)
    return Quaternion(a, p)


def quanternion_log(q: Quaternion) -> Quaternion:
    """Computes :math:`\\log(p)`

    Parameters
    ----------
    q : Quaternion

    Returns
    -------
    Quaternion
    """
    r = abs(q)
    a = log(r)
    p = sgn(q.imag) * arg(q)
    return Quaternion(a, p)


def arg(q: Quaternion) -> Scalar:
    return arccos(q.real * (1 / abs(q)))


def sgn(v: Array) -> Array:
    eps = jnp.finfo(v.dtype).eps
    r = norm(v)
    return lax.cond(
        r <= eps,
        lambda: zeros_like(v),
        lambda: v / r,
    )


def quaternion_rotation(x: Vec, q: Quaternion) -> Array:
    """Computes the quaternion rotation :math:`q^{-1} p q`.

    Parameters
    ----------
    x : Vec
        3d Vector
    q : Quaternion

    Returns
    -------
    Array
    """
    x_arr = asarray(x)
    qinv = q.reciprocal()
    p = Quaternion(zeros((), dtype=x_arr.dtype), x_arr)
    return (qinv * p * q).imag


def quaternion_reflection(x: Vec, normal_vec: Vec) -> Array:
    """Computes the reflection over the plane normal to the given normal vecor.

    Parameters
    ----------
    x : Vec
    normal_vec : Vec

    Returns
    -------
    Array
    """
    x_arr = asarray(x)
    normal_vec = asarray(normal_vec)
    n = normal_vec / norm(normal_vec)
    z = zeros((), dtype=x_arr.dtype)
    p = Quaternion(z, x)
    q = Quaternion(z, n)
    return (q * p * q).imag


def from_euler_angles(angles: Vec) -> Quaternion:
    """Returns the unit quaternion of the euler angles
    :math:`\\phi_x,\\phi_y,\\phi_z`

    Parameters
    ----------
    angles : Vec
        3d vector of euler angles.
    Returns
    -------
    Quaternion
    """
    angles_arr = asarray(angles)
    angles_arr = angles_arr / 2
    s = sin(angles_arr)
    c = cos(angles_arr)
    a = prod(c) + prod(s)
    q = (
        s[0] * c[1] * c[2] - c[0] * s[1] * s[2],
        c[0] * s[1] * c[2] + s[0] * c[1] * s[2],
        c[0] * c[1] * s[2] - s[0] * s[1] * c[2],
    )
    return Quaternion(a, q)


def from_axis_angle(angle: Scalar, axis: Vec) -> Quaternion:
    """Returns the unit quaternion of the given angle-axis representation

    Parameters
    ----------
    angle : Scalar
    axis : Vec
        3d Vector of the rotation

    Returns
    -------
    Quaternion
    """
    axis_arr = asarray(axis)
    angle = angle / 2
    return Quaternion(cos(angle), axis_arr * sin(angle))
