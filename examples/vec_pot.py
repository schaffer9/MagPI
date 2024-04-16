from dataclasses import dataclass, field

from chex import Array
from jaxopt.linear_solve import solve_lu

from magpi.calc import curl, laplace
from magpi.domain import Hypercube
from magpi.integrate import QuadRule, gauss, integrate, simpson
from magpi.prelude import *
from magpi.r_fun import cuboid, translate


ChargeFn = Callable[..., Array]
Scalar = Array
ChargeTensor = tuple[Array, Array, Array, Array, Array, Array]
TargetTensor = tuple[Array, Array, Array, Array, Array, Array]
GradTargetTensor = TargetTensor


@dataclass(frozen=True, eq=False)
class Face:
    axis1: int
    axis2: int
    const_axis: int
    axis1_vals: Array
    axis2_vals: Array
    const_val: Scalar
    midpoints: Array


Faces = tuple[Face, ...]


@dataclass(frozen=True, eq=False)
class Cuboid:
    xs: Array
    ys: Array
    zs: Array
    _dom: Hypercube = field(repr=False, init=False)
    faces: Faces = field(repr=False, init=False)
    dimension = property(lambda _: 3)
    lb = property(lambda self: array(self._dom.lb))
    ub = property(lambda self: array(self._dom.ub))

    def __post_init__(self):
        def _validate(arr):
            assert len(arr.shape) == 1
            assert len(arr) > 1
            assert jnp.all(arr.sort() == arr)

        _validate(self.xs)
        _validate(self.ys)
        _validate(self.zs)
        lb = tree_map(float, (self.xs[0], self.ys[0], self.zs[0]))
        ub = tree_map(float, (self.xs[-1], self.ys[-1], self.zs[-1]))
        dom = Hypercube(lb, ub)
        object.__setattr__(self, "_dom", dom)
        faces = _faces(self)
        object.__setattr__(self, "faces", faces)

    def support(self) -> Array:
        return self._dom.support()

    def includes(self, samples: Array) -> Array:
        return self._dom.includes(samples)

    def transform(self, samples: Array) -> Array:
        return self._dom.transform(samples)

    def transform_bnd(self, samples: Array) -> Array:
        return self._dom.transform_bnd(samples)

    def normalize(self, sample: Array) -> Array:
        """Shifts the sample between [-1, 1].

        Parameters
        ----------
        sample : Array

        Returns
        -------
        Array
        """
        return 2 * (sample - self.lb) / (self.ub - self.lb) - 1

    def adf(self, x: Array) -> Array:
        side_lengths = self.ub - self.lb
        adf = cuboid(side_lengths)
        adf = translate(adf, self.lb)
        return adf(x)

    def _normal_vec(self, x: Array) -> Array:
        return -jacfwd(self.adf)(x)

    def normal_vec(self, x: Array) -> Array:
        if len(x.shape) == 1:
            return self._normal_vec(lax.stop_gradient(x))
        else:
            return vmap(self._normal_vec)(lax.stop_gradient(x))


def _faces(dom: Cuboid) -> Faces:
    def face(axis1, axis2, const_axis, axis1_vals, axis2_vals, const_val):
        m = _midpoints(axis1_vals, axis2_vals, const_axis, const_val)
        return Face(axis1, axis2, const_axis, axis1_vals, axis2_vals, const_val, m)

    return (
        face(0, 1, 2, dom.xs, dom.ys, dom.zs[0]),
        face(0, 1, 2, dom.xs, dom.ys, dom.zs[-1]),
        face(0, 2, 1, dom.xs, dom.zs, dom.ys[0]),
        face(0, 2, 1, dom.xs, dom.zs, dom.ys[-1]),
        face(1, 2, 0, dom.ys, dom.zs, dom.xs[0]),
        face(1, 2, 0, dom.ys, dom.zs, dom.xs[-1]),
    )


def _midpoints(a, b, const_axis, const_val):
    m1 = (a[1:] + a[:-1]) / 2
    m2 = (b[1:] + b[:-1]) / 2
    M = stack(meshgrid(m1, m2, indexing="ij"), axis=-1)
    return jnp.insert(M, const_axis, const_val, axis=-1)


def solve_A2(
    domain: Cuboid,
    A: Callable[..., Array],
    m: Callable[..., Array],
    args_A1: tuple = (),
    args_m: tuple = (),
) -> Callable[[TargetTensor, GradTargetTensor], tuple[Array, Array, Array]]:
    def g(y):
        _m = m(y, *args_m)
        n = domain.normal_vec(y)
        return jnp.cross(_m, n) + A(y, *args_A1)  #  - jacfwd(A1)(y, *args_A1) @ n

    c = charges(domain, g)

    def A2_curl_A2_grad_A2(z, grad_z):
        A2 = 1 / (4 * pi) * _solve_sufrace_integral_A2(c, z)
        curl_A2 = 1 / (4 * pi) * _solve_curl_sufrace_integral_curl_A2(c, grad_z)
        grad_A2 = 1 / (4 * pi) * _solve_grad_sufrace_integral_grad_A2(c, grad_z)
        return A2, curl_A2, grad_A2

    return A2_curl_A2_grad_A2


def mag_energy(
    domain: Cuboid,
    A1: Callable[..., Array],
    m: Callable[..., Array],
    X_Z_grad_Z: tuple[Array, ChargeTensor, GradTargetTensor],
    args_A1: tuple = (),
    args_m: tuple = (),
    stop_gradient: bool = True,
) -> Scalar:
    X, Z, grad_Z = X_Z_grad_Z
    lA1 = lambda x: domain.adf(x) * A1(x, *args_A1)

    def e_dom(x, z, grad_z):
        JA1 = jacfwd(lA1)(x)
        curl_A1 = curl(lA1)(x)
        A2, curl_A2, JA2 = solve_A2(domain, A1, m, args_A1, args_m)(z, grad_z)
        if stop_gradient:
            curl_A1 = lax.stop_gradient(curl_A1)
            curl_A2 = lax.stop_gradient(curl_A2)
            A2 = lax.stop_gradient(A2)
            JA2 = lax.stop_gradient(JA2)
        e = (
            norm(m(x)) ** 2
            + jnp.sum(JA1 * JA1)
            - 2 * dot(m(x), curl_A1 + curl_A2)
            - curl(m)(x) @ A2
            + m(x) @ curl_A2
            - laplace(lA1)(x) @ A2
            - jnp.sum(JA1 * JA2)
        )
        return e

    return mean(vmap(e_dom)(X, Z, grad_Z))


def _solve_sufrace_integral_A2(charges, target) -> Array:
    def _dot_sum(a, b):
        return jnp.sum(a * b[..., None], axis=(0, 1, 2))

    face_contributions = tree_map(lambda c, t: _dot_sum(c, t), charges, target)
    A2 = jnp.sum(array(face_contributions), axis=0)
    return A2


def _solve_curl_sufrace_integral_curl_A2(charges, target) -> Array:
    def _dot_sum(a, b):
        return jnp.sum(-jnp.cross(a, b), axis=(0, 1, 2))

    face_contributions = tree_map(lambda c, t: _dot_sum(c, t), charges, target)
    curl_A2 = jnp.sum(array(face_contributions), axis=0)
    return curl_A2


def _solve_grad_sufrace_integral_grad_A2(charges, target) -> Array:
    def _dot_sum(a, b):
        return jnp.tensordot(a, b, ((0, 1, 2), (0, 1, 2)))

    face_contributions = tree_map(lambda c, t: _dot_sum(c, t), charges, target)
    curl_A2 = jnp.sum(array(face_contributions), axis=0)
    return curl_A2


def surface_integral(
    domain: Cuboid,
    f: ChargeFn,
    *,
    subintervals: tuple[int, int, int] = (1, 1, 1),
    method: QuadRule = simpson,
):
    charges = tree_map(lambda face: charge_taylor_coefs2(face, f), domain.faces)

    @jit
    def _integrate_surface(target_point):
        return integrate_surface(
            target_point,
            domain,
            charges,
            subintervals=subintervals,
            method=method,
        )

    return _integrate_surface


def single_layer_potential(
    domain: Cuboid,
    f: ChargeFn,
    *,
    subintervals: tuple[int, int, int] = (1, 1, 1),
    method: QuadRule = simpson,
):
    charges = tree_map(lambda face: charge_taylor_coefs2(face, f), domain.faces)

    @jit
    def _integrate_surface(target_point):
        return (
            1
            / (4 * pi)
            * integrate_surface(
                target_point,
                domain,
                charges,
                subintervals=subintervals,
                method=method,
            )
        )

    return _integrate_surface


def charges(dom: Cuboid, f: ChargeFn) -> ChargeTensor:
    return tree_map(lambda face: charge_taylor_coefs2(face, f), dom.faces)


@partial(jit, static_argnames=("dom", "subintervals", "method"))
def integrate_surface(
    target_point: Array,
    dom: Cuboid,
    charges: ChargeTensor,
    *,
    subintervals: tuple[int, int, int] = (1, 1, 1),
    method: QuadRule = simpson,
) -> Scalar:
    t = surface_tensors(target_point, dom, subintervals, method=method)
    return solve_sufrace_integral(charges, t)


def solve_sufrace_integral(charges: ChargeTensor, target: TargetTensor) -> Array:
    # def cross_sum(a, b):
    #     return jnp.sum(-jnp.cross(a, b), axis=(0,1,2))
    def _dot_sum(a, b):
        return jnp.sum(a * b[..., None], axis=(0, 1, 2))

    # face_contributions = tree_map(lambda c, t: jnp.tensordot(c, t, 3), charges, target)
    face_contributions = tree_map(lambda c, t: _dot_sum(c, t), charges, target)
    return jnp.sum(array(face_contributions), axis=0)


def charge_taylor_coefs2(
    face: Face,
    f: ChargeFn,
) -> Array:
    def _f(x):
        grad_f = jacfwd(f)
        hessian_f = jacfwd(grad_f)
        g = grad_f(x)
        H = hessian_f(x)
        i, j = face.axis1, face.axis2
        coefs = (
            f(x),
            g[..., i],
            g[..., j],
            1 / 2 * H[..., i, i],
            H[..., i, j],
            1 / 2 * H[..., j, j],
        )
        return stack(coefs)

    return jnp.apply_along_axis(_f, -1, face.midpoints)


def _integrand(y, x, center, face: Face):
    assert y.shape == (2,)
    assert x.shape == (3,)
    assert center.shape == (3,)

    _y = zeros_like(x)
    _y = _y.at[face.axis1].set(y[0])
    _y = _y.at[face.axis2].set(y[1])
    _y = _y.at[face.const_axis].set(face.const_val)
    a0 = 1.0
    a1 = _y[face.axis1] - center[face.axis1]
    a2 = _y[face.axis2] - center[face.axis2]
    a3 = a1**2
    a4 = a1 * a2
    a5 = a2**2
    return stack([a0, a1, a2, a3, a4, a5], axis=-1) / norm(x - _y)


def integrate_face(
    face: Face,
    target_point: Scalar,
    subintervals: int,
    method: QuadRule,
) -> Array:
    dom1 = stack([face.axis1_vals[:-1], face.axis1_vals[1:]], axis=-1)
    dom2 = stack([face.axis2_vals[:-1], face.axis2_vals[1:]], axis=-1)
    centers1 = (dom1[..., 0] + dom1[..., 1]) / 2
    centers2 = (dom2[..., 0] + dom2[..., 1]) / 2

    def _int(int_dom1, int_dom2, center1, center2):
        int_dom1 = linspace(int_dom1[0], int_dom1[-1], subintervals + 1)
        int_dom2 = linspace(int_dom2[0], int_dom2[-1], subintervals + 1)
        center = zeros((3,))
        center = center.at[face.axis1].set(center1)
        center = center.at[face.axis2].set(center2)
        center = center.at[face.const_axis].set(face.const_val)
        return integrate(
            _integrand,
            [int_dom1, int_dom2],
            target_point,
            center,
            face,
            method=method,
        )

    return vmap(vmap(_int, (None, 0, None, 0)), (0, None, 0, None))(
        dom1, dom2, centers1, centers2
    )


@partial(jit, static_argnames=("domain", "subintervals", "method"))
def surface_tensors(
    target_point: Array,
    domain: Cuboid,
    subintervals: int,
    method: QuadRule = simpson,
):
    def _face_tensor(face):
        return integrate_face(face, target_point, subintervals, method)

    return tree_map(_face_tensor, domain.faces)


@partial(jit, static_argnames=("domain", "subintervals", "method"))
def surface_tensors_grad(
    target_point: Array,
    domain: Cuboid,
    subintervals: int,
    method: QuadRule = simpson,
):
    compute_tensors = lambda x: surface_tensors(x, domain, subintervals, method)
    return jacfwd(compute_tensors)(target_point)


def unit_vec(x):
    return x / norm(x, axis=-1, keepdims=True)


def to_skew_simmetric_matrix(x):
    S = zeros((3, 3))
    S = S.at[1, 0].set(x[2])
    S = S.at[2, 0].set(-x[1])
    S = S.at[2, 1].set(x[0])
    S = S - S.T
    return S


def cayley_rotation(p, x):
    assert p.shape[0] == 3, f"{p.shape}"
    Q = to_skew_simmetric_matrix(p)
    I = jnp.eye(3)
    return jnp.linalg.inv(I - Q) @ (I + Q) @ x
