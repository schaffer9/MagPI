from .prelude import *
from .quaternions import (
    quaternion_rotation, 
    from_euler_angles, 
    from_axis_angle,
)


Scalar = Array | float | int
Vec = Array | list[Scalar] | Scalar | tuple[Scalar, ...]
Vec2d = Array | tuple[Scalar, Scalar]
Vec3d = Array | tuple[Scalar, Scalar, Scalar]


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

    def conjunction(self, adf1: ADF, adf2: ADF) -> ADF:
        raise NotADirectoryError
    
    def disjunction(self, adf1: ADF, adf2: ADF) -> ADF:
        raise NotADirectoryError
    
    def negate(self, adf: ADF) -> ADF:
        return lambda x: -adf(x)

    def union(self, adf1: ADF, adf2: ADF) -> ADF:
        return self.disjunction(adf1, adf2)
    
    def intersection(self, adf1: ADF, adf2: ADF) -> ADF:
        return self.conjunction(adf1, adf2)

    def equivalence(self, adf1: ADF, adf2: ADF) -> ADF:
        return self.negate(self.xor(adf1, adf2))

    def implication(self, adf1: ADF, adf2: ADF) -> ADF:
        return self.union(self.negate(adf1), adf2)

    def difference(self, adf1: ADF, adf2: ADF) -> ADF:
        return self.intersection(adf1, self.negate(adf2))
    
    def xor(self, adf1: ADF, adf2: ADF) -> ADF:
        adf1, adf2 = self.union(adf1, adf2), self.negate(self.intersection(adf1, adf2))
        return self.intersection(adf1, adf2)
    

class RAlpha(RFun):
    def __init__(self, alpha: Callable[[Scalar, Scalar], Scalar] = 0):
        self.alpha = alpha

    def conjunction(self, adf1: ADF, adf2: ADF) -> ADF:
        def op(a, b):
            alpha = self.alpha(a, b)
            return 1 / (1 + alpha) * (a + b - sqrt(a ** 2 + b ** 2 - 2 * alpha * a * b))
        
        return lambda x: op(adf1(x), adf2(x))
    
    def disjunction(self, adf1: ADF, adf2: ADF) -> ADF:
        def op(a, b):
            alpha = self.alpha(a, b)
            return 1 / (1 + alpha) * (a + b + sqrt(a ** 2 + b ** 2 - 2 * alpha * a * b))
        
        return lambda x: op(adf1(x), adf2(x))


class RAlphaM(RFun):
    def __init__(self, m: Scalar, alpha: Callable[[Scalar, Scalar], Scalar] = 0):
        self.m = m
        self.alpha = alpha

    def conjunction(self, adf1: ADF, adf2: ADF) -> ADF:
        def op(a, b):
            alpha = self.alpha(a, b)
            r = a ** 2 + b ** 2
            return 1 / (1 + alpha) * (a + b - sqrt(r - 2 * alpha * a * b)) * r ** (self.m / 2)
        
        return lambda x: op(adf1(x), adf2(x))
    
    def disjunction(self, adf1: ADF, adf2: ADF) -> ADF:
        def op(a, b):
            alpha = self.alpha(a, b)
            r = a ** 2 + b ** 2
            return 1 / (1 + alpha) * (a + b + sqrt(r - 2 * alpha * a * b)) * r ** (self.m / 2)
        
        return lambda x: op(adf1(x), adf2(x))

class RP(RFun):
    def __init__(self, p: int):
        assert p % 2 == 0, "`p` must be an even integer"
        self.p = p

    def conjunction(self, adf1: ADF, adf2: ADF) -> ADF:
        def op(a, b):
            return a + b - (a ** self.p + b ** self.p) ** (1 / self.p)
        
        return lambda x: op(adf1(x), adf2(x))
    
    def disjunction(self, adf1: ADF, adf2: ADF) -> ADF:
        def op(a, b):
            return a + b + (a ** self.p + b ** self.p) ** (1 / self.p)
        
        return lambda x: op(adf1(x), adf2(x))

r1 = RAlpha(lambda a, b: 1)  # min, max
r0 = RAlpha(lambda a, b: 0)  # analytic everywhere but the origin and normalized to first order
rp2 = RP(2)  # same as r0
rp4 = RP(4)  # analytic everywhere and normalized to 3rd order.


def cuboid(l: Vec, centering: bool = False, normalize: int = 1) -> ADF:
    assert normalize % 2 == 1, "Only odd degrees of normalization allowed for cuboid"
    l = array(l)
    
    if centering:
        lb = -l / 2
        ub = l / 2
    else:
        lb = zeros_like(l)
        ub = l
    
    # use a RP function to compute the intersection for all 6 sides
    p = normalize + 1
    _intersection = compose(lambda a, b: a + b - (a ** p + b ** p) ** (1 / p))
    @_intersection
    def adf(x):
        a = (ub - x).ravel()
        b = (x - lb).ravel()
        return concatenate([a, b])
    
    return adf


def cube(l: Scalar, centering: bool = False, normalize: int = 1) -> RFun:
    return cuboid(l, centering, normalize)


def sphere(r: Scalar) -> ADF:
    return lambda x: (r ** 2 - norm(x) ** 2) / (2 * r)

def cylinder(r: Scalar, h: None | Scalar = None):
    s = sphere(r)
    adf = lambda x: s(x[:2])
    if h is not None:
        _cube = cube(h, centering=False)
        def cut_fn(x):
            x = x.at[:2].set(x[:2] - r)
            return _cube(x)
        adf = r0.intersection(cut_fn, adf)
    
    return adf
        

def compose(func: Callable[[Scalar, Scalar], Scalar]) -> Callable[..., ADF]:
    def composition(*adf):
        def _adf(x):
            d = concatenate(tree_leaves(tree_map(lambda df: df(x).ravel(), adf)))
            return reduce(func, d)
        return _adf
    return composition


def translate(adf: ADF, y: Vec) -> ADF:
    y = array(y)
    return lambda x: adf(x - y)


def scale(adf: ADF, scaling_factor: Scalar) -> ADF:
    scaling_factor = array(scaling_factor).ravel()
    assert scaling_factor.shape == (1,), "`scaling_factor` must be a scalar to preserve normalization"
    scaling_factor = scaling_factor[0]
    return lambda x: adf(x / scaling_factor) * scaling_factor    


def scale_without_normalization(adf, scaling_factor) -> ADF:
    scaling_factor = array(scaling_factor).ravel()
    return lambda x: adf(x / scaling_factor)


def rotate2d(adf: ADF, angle: Scalar, o: Vec2d = (0., 0.)) -> ADF:
    o = array(o)
    _adf = translate(adf, -o)
    M = array([[cos(angle), -sin(angle)], 
              [sin(angle), cos(angle)]])
    def rot_op(x):
        assert x.shape == (2,), f"Cannot rotate vector of size {x.shape} in 2d. Please pass a 2d vector."
        return _adf(M @ x)
    
    return translate(rot_op, o)
    

def rotate3d(adf: ADF, 
             angle: Scalar | Vec3d, 
             rot_axis: None | Vec3d = None, 
             o: Vec3d = (0., 0., 0.)) -> ADF:
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
    return translate(rot_op, o)


def reflect(adf: ADF,
            normal_vec: Vec2d,
            o: Vec = 0.) -> ADF:
    o, n = array(o), array(normal_vec)
    _adf = translate(adf, -o)
    def ref_op(x):
        x = x - 2 * (x @ n) / (norm(n) ** 2) * n
        return _adf(x)
    return translate(ref_op, o)


def project(adf: ADF,
            normal_vec: Vec2d,
            o: Vec = 0.) -> ADF:
    o, n = array(o), array(normal_vec)
    _adf = translate(adf, -o)
    def proj_op(x):
        x = x - (x @ n) / (norm(n) ** 2) * n
        return _adf(x)
    return translate(proj_op, o)


def normalize_1st_order(adf: ADF) -> ADF:
    df = jacfwd(adf)
    def normalize(x):
        y = adf(x)
        dy = df(x)
        return y / sqrt(y ** 2 * norm(dy) ** 2)
    return normalize