from .prelude import *
from .quaternions import (
    quaternion_rotation, 
    from_euler_angles, 
    from_axis_angle,
)


Scalar = Array | float
Vec = Array | list[float] | float
Vec2d = Array | tuple[float, float]
Vec3d = Array | tuple[float, float, float]


class ADF(T.Protocol):
    """
    ADF stands for Approximate distance function. It is
    positive inside the respective domain, zero on the boundary 
    and negative outside the domain. When normalized to first order
    the normal derivative has a magnitude of one everywhere on the
    boundary. Higher order normalization yields a function
    where higher order normal derivatives are zero.
    """
    def __call__(self, x: Array) -> Scalar: ...


class RFun:
    conjunction = NotImplemented
    disjunction = NotImplemented

    def __init__(self, f: ADF):
        self._f = f
        
    def __call__(self, x: Array) -> Scalar:
        return self._f(x)

    def __and__(self, other: 'RFun') -> 'RFun':
        def op(x):
            return self.conjunction(self(x), other(x))
        return self.__class__(op)

    def __or__(self, other: 'RFun') -> 'RFun':
        def op(x):
            return self.disjunction(self(x), other(x))
        return self.__class__(op)

    def __xor__(self, other: 'RFun') -> 'RFun':
        return (self | other) & ~(self & other)

    def __invert__(self) -> 'RFun':
        def op(x):
            return -self(x)
        return self.__class__(op)
    
    def __add__(self, other: Union['RFun', Scalar]) -> 'RFun':
        def op(x):
            if isinstance(other, self.__class__):
                return self(x) + other(x)
            else:
                return self(x) + other
        return self.__class__(op)
    
    def __radd__(self, other: Union['RFun', Scalar]) -> 'RFun':
        return self + other
    
    def __sub__(self, other: Union['RFun', Scalar]) -> 'RFun':
        return self + (-other)
    
    def __rsub__(self, other: Union['RFun', Scalar]) -> 'RFun':
        return self - other
    
    def __mul__(self, other: Union['RFun', Scalar]) -> 'RFun':
        def op(x):
            if isinstance(other, self.__class__):
                return self(x) * other(x)
            else:
                return self(x) * other
        return self.__class__(op)
    
    def __rmul__(self, other: Union['RFun', Scalar]) -> 'RFun':
        return self * other
    
    def __truediv__(self, other: Union['RFun', Scalar]) -> 'RFun':
        def op(x):
            if isinstance(other, self.__class__):
                return self(x) / other(x)
            else:
                return self(x) / other
        return self.__class__(op)
    
    def __rtruediv__(self, other: Union['RFun', Scalar]) -> 'RFun':
        def op(x):
            return other / self(x)
        return self.__class__(op)

    def normalize_1st_order(self) -> 'RFun':
        def op(x):
            return self(x) / sqrt(self(x) ** 2 + norm(grad(self)(x)) ** 2)
        return self.__class__(op)

    def translate(self, y: Vec) -> 'RFun':
        return translate(self, y)
    
    def scale(self, scaling_factor: Scalar) -> 'RFun':
        return scale(self, scaling_factor)


def rp_conjunction(a, b, p=2):
    return a + b - (a ** p + b ** p) ** (1 / p)


def rp_disjunction(a, b, p=2):
    return a + b + (a ** p + b ** p) ** (1 / p)


class Rp2Fun(RFun):
    """
    """
    conjunction = partial(rp_conjunction, p=2)
    disjunction = partial(rp_disjunction, p=2)


class Rp4Fun(RFun):
    """
    """
    conjunction = partial(rp_conjunction, p=4)
    disjunction = partial(rp_disjunction, p=4)
    

def cuboid(l: Vec, centering: bool = False, r_func: type[RFun] = Rp2Fun) -> RFun:
    l = array(l)
    
    if centering:
        lb = -l / 2
        ub = l / 2
    else:
        lb = zeros_like(l)
        ub = l
    
    @compose(r_func.conjunction)
    def adf(x):
        a = ub - x
        b = x - lb
        return concatenate([a, b])
    
    return r_func(adf)


def cube(l: Scalar, dim: int=3, centering: bool = False, r_func: type[RFun] = Rp2Fun) -> RFun:
    return cuboid([l] * dim, centering, r_func)


def sphere(r: Scalar, origin: Vec = 0., r_func: type[RFun] = Rp2Fun) -> RFun:
    def adf(x):
        return (r ** 2 - norm(x - origin) ** 2) / (2 * r)
    return r_func(adf)


def compose(func):
    def composition(*adf):
        def _adf(x):
            d = concatenate(tree_leaves(tree_map(lambda df: df(x).ravel(), adf)))
            return reduce(func, d)
        return _adf
    return composition


def translate(adf: RFun, y: Vec) -> 'RFun':
    y = array(y)
    def op(x):
        return adf(x - y)
    return adf.__class__(op)


def scale(adf: RFun, scaling_factor: Scalar) -> RFun:
    scaling_factor = array(scaling_factor).ravel()
    assert scaling_factor.shape == (1,), "`scaling_factor` must be a scalar to preserve normalization"
    scaling_factor = scaling_factor[0]
    def op(x):
        return adf(x / scaling_factor) * scaling_factor    
    return adf.__class__(op)


def rotate2d(adf: RFun, angle: Scalar, o: Vec2d = (0., 0.)) -> RFun:
    o = array(o)
    _adf = translate(adf, -o)
    M = array([[cos(angle), -sin(angle)], 
              [sin(angle), cos(angle)]])
    def rot_op(x):
        assert x.shape == (2,), f"Cannot rotate vector of size {x.shape} in 2d. Please pass a 2d vector."
        return _adf(M @ x)
    return translate(adf.__class__(rot_op), o)
    

def rotate3d(adf: RFun, 
             angle: Scalar | Vec3d, 
             rot_axis: None | Vec3d = None, 
             o: Vec3d = (0., 0., 0.)) -> RFun:
    o = array(o)
    _adf = translate(adf, -o)

    angle = array(angle)
    if angle.shape == ():
        if rot_axis is None:
            msg = "If only the angle is specified, the rotation axis must be provided"
            raise ValueError(msg)
        rot_quaternion = from_axis_angle(angle, rot_axis)
    elif angle.shape == (3,):
        if rot_axis is not None:
            raise ValueError("If Euler angles are given, the `rot_axis` must be `None`")
        rot_quaternion = from_euler_angles(angle)
    else:
        raise ValueError("Provide axis-angle representation or Euler angles.")

    def rot_op(x):
        assert x.shape == (3,), f"Cannot rotate vector of size {x.shape} in 3d. Please pass a 3d vector."
        x = quaternion_rotation(x, rot_quaternion)
        return _adf(x)
    return translate(adf.__class__(rot_op), o)


def reflect(adf: RFun,
            normal_vec: Vec2d,
            o: Vec = 0.) -> RFun:
    o = array(o)
    n = array(normal_vec)
    n = n / norm(n)
    _adf = translate(adf, -o)
    def ref_op(x):
        x = x - 2 * (x @ n) / norm(n) * n
        return _adf(x)
    return translate(adf.__class__(ref_op), o)


def union(adf1: RFun, adf2: RFun) -> RFun:
    return adf1 | adf2

def intersection(adf1: RFun, adf2: RFun) -> RFun:
    return adf1 & adf2

def equivalence(adf1: RFun, adf2: RFun) -> RFun:
    return ~(adf1 ^ adf2)

def material_conditional(adf1: RFun, adf2: RFun) -> RFun:
    return ~adf1 | adf2

def difference(adf1: RFun, adf2: RFun) -> RFun:
    return adf1 & ~adf2