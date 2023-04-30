from .prelude import *


Scalar = Array


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


class RFunc:
    conjunction = NotImplemented
    disjunction = NotImplemented

    def __init__(self, f: ADF):
        self._f = f
        
    def __call__(self, x: Array) -> Scalar:
        return self._f(x)

    def __and__(self, other: 'RFunc') -> 'RFunc':
        def op(x):
            return self.conjunction(self(x), other(x))
        return self.__class__(op)

    def __or__(self, other: 'RFunc') -> 'RFunc':
        def op(x):
            return self.disjunction(self(x), other(x))
        return self.__class__(op)

    def __xor__(self, other: 'RFunc') -> 'RFunc':
        return (self | other) & ~(self & other)

    def __truediv__(self, other: 'RFunc') -> 'RFunc':
        return self & ~other

    def __invert__(self) -> 'RFunc':
        def op(x):
            return -self(x)
        return self.__class__(op)
          
    def normalize_1st_order(self) -> 'RFunc':
        def op(x):
            return self(x) / sqrt(self(x) ** 2 + norm(grad(self)(x)) ** 2)
        return self.__class__(op)

    def translate(self, y: Array) -> 'RFunc':
        return translate(self, y)
    
    def scale(self, scaling_factor: float | Array) -> 'RFunc':
        return scale(self, scaling_factor)


def rp_conjunction(a, b, p=2):
    return a + b - (a ** p + b ** p) ** (1 / p)


def rp_disjunction(a, b, p=2):
    return a + b + (a ** p + b ** p) ** (1 / p)


class Rp2Func(RFunc):
    """
    """
    conjunction = partial(rp_conjunction, p=2)
    disjunction = partial(rp_disjunction, p=2)
    

def cuboid(l: float|Array|list[float], centering: bool = False, r_func: type[RFunc] = Rp2Func) -> RFunc:
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


def cube(l: float, dim: int=3, centering: bool = False, r_func: type[RFunc] = Rp2Func) -> RFunc:
    return cuboid([l] * dim, centering, r_func)


def sphere(r: float, origin: float | Array = 0., r_func: type[RFunc] = Rp2Func) -> RFunc:
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


def translate(adf: RFunc, y: Array) -> 'RFunc':
    def op(x):
        return adf(x - y)
    return adf.__class__(op)


def scale(adf: RFunc, scaling_factor: float | Array | list[float]) -> 'RFunc':
    scaling_factor = array(scaling_factor)
    def op(x):
        return adf(x / scaling_factor)
    
    return adf.__class__(op).normalize_1st_order()

def rotate2d(adf: RFunc, angle, axis):
    pass

def rotate3d(adf: RFunc, angle, axis):
    pass

def union(adf1: RFunc, adf2: RFunc) -> RFunc:
    return adf1 | adf2

def intersection(adf1: RFunc, adf2: RFunc) -> RFunc:
    return adf1 & adf2

def equivalence(adf1: RFunc, adf2: RFunc) -> RFunc:
    return ~(adf1 ^ adf2)

def material_conditional(adf1: RFunc, adf2: RFunc) -> RFunc:
    return ~adf1 | adf2
