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

_annotation = """
ADF stands for Approximate distance function. It is
positive inside the respective domain, zero on the boundary
and negative outside the domain. When normalized to first order
the normal derivative has a magnitude of one everywhere on the
boundary. Higher order normalization yields a function
where higher order normal derivatives are zero.
"""
ADF = T.Annotated[Callable[[Array | Scalar], Scalar], _annotation]


class RFun:
    """
    Implements the basic set theoretic operations for a system of R-Functions.
    In essence, only `conjugation` and `disjunction` needs to be implemented.
    """

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
    """
    Implements the system :math:`R_\\alpha` for some value of `alpha` in (-1, 1].

    Parameters
    ----------
    alpha : Scalar
    """

    def __init__(self, alpha: Scalar):
        assert -1.0 < alpha <= 1.0
        self.alpha = alpha

    def conjunction(self, adf1: ADF, adf2: ADF) -> ADF:
        def op(a, b):
            return (
                1
                / (1 + self.alpha)
                * (a + b - sqrt(a**2 + b**2 - 2 * self.alpha * a * b))
            )

        return lambda x: op(adf1(x), adf2(x))

    def disjunction(self, adf1: ADF, adf2: ADF) -> ADF:
        def op(a, b):
            return (
                1
                / (1 + self.alpha)
                * (a + b + sqrt(a**2 + b**2 - 2 * self.alpha * a * b))
            )

        return lambda x: op(adf1(x), adf2(x))


class RAlphaM(RFun):
    """
    Implements the system :math:`R_\\alpha^m` for some value of `alpha` in (-1, 1].
    Note that this system is not normalized.

    Parameters
    ----------
    m: Scalar
    alpha : Scalar
    """

    def __init__(self, m: Scalar, alpha: Scalar):
        assert -1.0 < alpha <= 1.0
        self.m = m
        self.alpha = alpha

    def conjunction(self, adf1: ADF, adf2: ADF) -> ADF:
        def op(a, b):
            r = a**2 + b**2
            return (
                1
                / (1 + self.alpha)
                * (a + b - sqrt(r - 2 * self.alpha * a * b))
                * r ** (self.m / 2)
            )

        return lambda x: op(adf1(x), adf2(x))

    def disjunction(self, adf1: ADF, adf2: ADF) -> ADF:
        def op(a, b):
            r = a**2 + b**2
            return (
                1
                / (1 + self.alpha)
                * (a + b + sqrt(r - 2 * self.alpha * a * b))
                * r ** (self.m / 2)
            )

        return lambda x: op(adf1(x), adf2(x))


class RP(RFun):
    """
    Implements the system :math:`R_p` for some even integer value `p`.
    This system is normalized to the order `p-1`.

    Parameters
    ----------
    p : int
    """

    def __init__(self, p: int):
        assert p % 2 == 0, "`p` must be an even integer"
        self.p = p

    def conjunction(self, adf1: ADF, adf2: ADF) -> ADF:
        def op(a, b):
            return a + b - (a**self.p + b**self.p) ** (1 / self.p)

        return lambda x: op(adf1(x), adf2(x))

    def disjunction(self, adf1: ADF, adf2: ADF) -> ADF:
        def op(a, b):
            return a + b + (a**self.p + b**self.p) ** (1 / self.p)

        return lambda x: op(adf1(x), adf2(x))


class RhoBlending(RFun):
    """Blending R-function which smoothes sharp corners and edges [1]_.

    Parameters
    ---------
    rho : float
        smoothing factor

    Notes
    -----
    .. [1] Shapiro, Vadim. "Semi-analytic geometry with R-functions."
       ACTA numerica 16 (2007): 239-303.
    """

    def __init__(self, rho: float):
        self.rho = rho

    def conjunction(self, adf1: ADF, adf2: ADF) -> ADF:
        def op(a, b):
            s = a**2 + b**2 - self.rho**2
            return (
                a
                + b
                - sqrt(a**2 + b**2 + 1 / (8 * self.rho) * s * (s - jnp.abs(s)))
            )

        return lambda x: op(adf1(x), adf2(x))

    def disjunction(self, adf1: ADF, adf2: ADF) -> ADF:
        def op(a, b):
            s = a**2 + b**2 - self.rho**2
            return (
                a
                + b
                + sqrt(a**2 + b**2 + 1 / (8 * self.rho) * s * (s - jnp.abs(s)))
            )

        return lambda x: op(adf1(x), adf2(x))


r1 = RAlpha(1.0)  # min, max
r0 = RAlpha(0.0)  # analytic everywhere but the origin and normalized to first order
rp2 = RP(2)  # same as r0
rp4 = RP(4)  # analytic everywhere and normalized to 3rd order.


def cuboid(edge_lengths: Vec, centering: bool = False, normalize: int = 1) -> ADF:
    """
    Returns the ADF of a cuboid.

    Parameters
    ----------
    edge_lengths : Vec
        geometry of the cuboid. The lenght of the vector determines the dimension.
    centering : bool, optional
        centers the cuboid at the origin, by default False
    normalize : int, optional
        normalization degree of the ADF, by default 1

    Returns
    -------
    ADF
    """
    assert normalize % 2 == 1, "Only odd degrees of normalization allowed for cuboid"
    _edge_lengths = asarray(edge_lengths)

    if centering:
        lb = -_edge_lengths / 2
        ub = _edge_lengths / 2
    else:
        lb = zeros_like(_edge_lengths)
        ub = _edge_lengths

    # use a RP function to compute the intersection for all 6 sides
    p = normalize + 1
    _intersection = compose(lambda a, b: a + b - (a**p + b**p) ** (1 / p))

    @_intersection
    def adf(x):
        a = (ub - x).ravel()
        b = (x - lb).ravel()
        return concatenate([a, b])

    return adf


def cube(edge_lenght: Scalar, centering: bool = False, normalize: int = 1) -> ADF:
    """
    Return the ADF of a cube.

    Parameters
    ----------
    edge_lenght : Scalar
        edge length of the cube
    centering : bool, optional
        centers the cuboid at the origin, by default False
    normalize : int, optional
        normalization degree of the ADF, by default 1

    Returns
    -------
    ADF
    """
    return cuboid(edge_lenght, centering, normalize)


def sphere(r: Scalar) -> ADF:
    """
    Returns the ADF of a sphere which is normalized to first order.
    The dimension is arbitrary.

    Parameters
    ----------
    r : Scalar
        radius

    Returns
    -------
    ADF
    """
    return lambda x: (r**2 - norm(x) ** 2) / (2 * r)


def cylinder(r: Scalar) -> ADF:
    """
    1st order ADF of a cylinder of infinte length.
    The base of the cylinder lies in the first two dimensions of the
    input.

    Parameters
    ----------
    r : Scalar
        radius

    Returns
    -------
    ADF
    """
    s = sphere(r)
    return lambda x: s(asarray(x)[:2])


def ellipsoid(axes_lengths: Vec) -> ADF:
    """
    1st order ADF of a ellipsoid.

    Parameters
    ----------
    axes_lengths : Vec
        length of the vector determines the dimension

    Returns
    -------
    ADF
    """
    adf = sphere(1.0)
    adf = scale_without_normalization(adf, axes_lengths)
    adf = normalize_1st_order(adf)
    return adf


def compose(func: Callable[[Scalar, Scalar], Scalar]) -> Callable[..., ADF]:
    def composition(*adf):
        def _adf(x):
            d = concatenate(tree_leaves(tree_map(lambda df: df(x).ravel(), adf)))
            return reduce(func, d)

        return _adf

    return composition


def translate(adf: ADF, y: Vec) -> ADF:
    """
    Translates the ADF by the vector y.

    Parameters
    ----------
    adf : ADF
    y : Vec

    Returns
    -------
    ADF
    """
    _y = asarray(y)
    return lambda x: adf(x - _y)


def scale(adf: ADF, scaling_factor: Scalar) -> ADF:
    """
    Scales the ADF by the given `scaling_factor`.
    First order normalization is preserved.

    Parameters
    ----------
    adf : ADF
    scaling_factor : Scalar

    Returns
    -------
    ADF
    """
    _scaling_factor = asarray(scaling_factor).ravel()
    assert _scaling_factor.shape == (
        1,
    ), "`scaling_factor` must be a scalar to preserve normalization"
    _scaling_factor = _scaling_factor[0]
    return lambda x: adf(x / _scaling_factor) * _scaling_factor


def scale_without_normalization(adf: ADF, scaling_factor: Vec | Scalar) -> ADF:
    """
    Scales the ADF without normalization. This allows
    different scaling factors for each dimension but does
    not preserve normalization. First order normalization can be
    estabished by `normalize_1st_order`.

    Parameters
    ----------
    adf : ADF
    scaling_factor : Vec | Scalar

    Returns
    -------
    ADF
    """
    _scaling_factor = asarray(scaling_factor).ravel()
    return lambda x: adf(x / _scaling_factor)


def rotate2d(adf: ADF, angle: Scalar, o: Vec2d = (0.0, 0.0)) -> ADF:
    """
    Rotates the given 2d ADF by some `angle` around the point `o`.

    Parameters
    ----------
    adf : ADF
    angle : Scalar
    o : Vec2d, optional
        by default (0.0, 0.0)

    Returns
    -------
    ADF
    """
    _o = asarray(o)
    _angle = asarray(angle)
    _adf = translate(adf, -_o)
    M = array(
        [[cos(_angle), sin(_angle)], [-sin(_angle), cos(_angle)]], dtype=_angle.dtype
    )

    def rot_op(x):
        msg = f"Cannot rotate vector of size {x.shape} in 2d. Please pass a 2d vector."
        assert x.shape == (2,), msg
        return _adf(M @ x)

    return translate(rot_op, _o)


def rotate3d(
    adf: ADF,
    angle: Scalar | Vec3d,
    rot_axis: None | Vec3d = None,
    o: Vec3d = (0.0, 0.0, 0.0),
) -> ADF:
    """
    Rotates the 3d ADF by some angle around the rotation axis `rot_axis` or
    if a three euler angles are provided around the point `o`.

    Parameters
    ----------
    adf : ADF
    angle : Scalar | Vec3d
    rot_axis : None | Vec3d, optional
        by default None
    o : Vec3d, optional
        by default (0.0, 0.0, 0.0)

    Returns
    -------
    ADF

    Raises
    ------
    ValueError
    """
    _o = asarray(o)
    _adf = translate(adf, -_o)

    _angle = -asarray(angle)
    if _angle.shape == ():
        if rot_axis is None:
            msg = "If only the angle is specified, the rotation axis must be provided"
            raise ValueError(msg)
        rot_quaternion = from_axis_angle(_angle, rot_axis)
    elif _angle.shape == (3,):
        if rot_axis is not None:
            raise ValueError("If Euler angles are given, the `rot_axis` must be `None`")
        rot_quaternion = from_euler_angles(_angle)
    else:
        raise ValueError("Provide axis-angle representation or Euler angles.")

    def rot_op(x):
        msg = f"Cannot rotate vector of size {x.shape} in 3d. Please pass a 3d vector."
        assert x.shape == (3,), msg
        x = quaternion_rotation(x, rot_quaternion)
        return _adf(x)

    return translate(rot_op, _o)


def reflect(adf: ADF, normal_vec: Vec, o: Vec = 0.0) -> ADF:
    """
    Reflects the ADF along the provided normal vector of the reflection plane with origin `o`.


    Parameters
    ----------
    adf : ADF
    normal_vec : Vec
    o : Vec, optional
        by default 0.0

    Returns
    -------
    ADF
    """
    _o, _n = asarray(o), asarray(normal_vec)
    _adf = translate(adf, -_o)

    def ref_op(x):
        x = x - 2 * (x @ _n) / (norm(_n) ** 2) * _n
        return _adf(x)

    return translate(ref_op, _o)


def project(adf: ADF, normal_vec: Vec, o: Vec = 0.0) -> ADF:
    """
    Projects the ADF onto the plane devined by the normal vector `normal_vec` and `o`.

    Parameters
    ----------
    adf : ADF
    normal_vec : Vec
    o : Vec, optional
        by default 0.0

    Returns
    -------
    ADF
    """
    _o, _n = asarray(o), asarray(normal_vec)
    _adf = translate(adf, -_o)

    def proj_op(x):
        x = x - (x @ _n) / (norm(_n) ** 2) * _n
        return _adf(x)

    return translate(proj_op, o)


def revolution(adf: ADF, axis: int = 0) -> ADF:
    """Creates the body of revolution from a 2d shape.

    Parameters
    ----------
    adf : ADF
    axis : int, optional
        0 for revolution around x axis or 1 for y axis, by default 0
    """

    def rev_op(x):
        assert x.shape == (3,), "Revolution only supported in 3d."
        if axis == 0:
            r = sqrt(x[1] ** 2 + x[2] ** 2)
            return adf(jnp.stack([x[0], r]))
        elif axis == 1:
            r = sqrt(x[0] ** 2 + x[2] ** 2)
            return adf(jnp.stack([r, x[1]]))
        else:
            msg = "`axis` must either be 0 for revolution around x "
            msg += "or 1 for revolution around y."
            raise ValueError(msg)

    return rev_op


def normalize_1st_order(adf: ADF) -> ADF:
    """
    Normalizes the ADF to first order. Note that the gradient
    cannot vanish on the boundary for this function to work.

    Parameters
    ----------
    adf : ADF

    Returns
    -------
    ADF
    """
    df = jacfwd(adf)

    def normalize(x):
        y = adf(x)
        dy = df(x)
        return y / sqrt(y**2 + norm(dy) ** 2)

    return normalize
