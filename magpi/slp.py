from typing import Callable, TypeVar, Any, overload, Literal
import math

from chex import ArrayTree

from .prelude import *
from .integrate import integrate_quad_rule, Integrand
from .calc import value_and_jacfwd
from .mesh import Mesh


T = TypeVar("T", bound=ArrayTree, covariant=True)

TriElement = Array
RefSpace = Array
PhySpace = Array
TriWeights = Array
TriNodes = Array
Sources = Array
Charges = Array
NormalVec = Array


@overload
def source_tensor_for_mesh(
    x: PhySpace,
    mesh: Mesh,
    weights: TriWeights,
    nodes: TriNodes,
    *,
    order: int = 2,
    compute_jacfwd: Literal[True] = True
) -> tuple[Sources, Sources]:
    ...


@overload
def source_tensor_for_mesh(
    x: PhySpace,
    mesh: Mesh,
    weights: TriWeights,
    nodes: TriNodes,
    *,
    order: int = 2,
    compute_jacfwd: Literal[False] = False
) -> Sources:
    ...


def source_tensor_for_mesh(
    x: PhySpace, mesh: Mesh, weights: TriWeights, nodes: TriNodes, *, order: int = 2, compute_jacfwd: bool = True
):
    def source_for_element(i):
        element = mesh.nodes[i]
        z = source_tensor(x, element, weights, nodes, order=order, compute_jacfwd=compute_jacfwd)
        return z

    if compute_jacfwd:
        z, dz = vmap(source_for_element)(mesh.sur_elements)
        return z, dz
    else:
        z = vmap(source_for_element)(mesh.sur_elements)
        z = z.reshape(-1)
        return z


def charge_tensor_for_mesh(f: Callable[..., Array], mesh: Mesh, *args: Any, order: int = 2, **kwargs: Any) -> Charges:
    def inner(i):
        element = mesh.nodes[i]
        c = taylor_coeffs(f, element, *args, order=order, **kwargs)
        return c

    c = vmap(inner)(mesh.sur_elements)
    return c


def from_ref_element(x: RefSpace, element: TriElement) -> PhySpace:
    v0, v1, v2 = element
    y = (1 - x[..., 0] - x[..., 1]) * v0 + x[..., 0] * v1 + x[..., 1] * v2
    return y


def integrate_surface_element(
    f: Integrand[T], element: TriElement, weights: TriWeights, nodes: TriNodes, *args: Any, **kwargs: Any
) -> T:
    parametrization = lambda x: from_ref_element(x, element)

    def integrand(x):
        r, J = value_and_jacfwd(parametrization)(x)
        msg = "Please provide a surface parametrization from [0,1]² -> R³"
        assert J.shape == (3, 2), msg
        ru = J[:, 0]
        rv = J[:, 1]
        z = norm(cross(ru, rv))
        return tree_map(lambda y: y * z, f(r, *args, **kwargs))

    return integrate_quad_rule(integrand, weights, nodes)


def center(element: TriElement) -> PhySpace:
    return from_ref_element(array([1 / 3, 1 / 3]), element)


@partial(jit, static_argnames=("order", "compute_jacfwd"))
def source_tensor(
    x: PhySpace,
    element: TriElement,
    weights: TriWeights,
    nodes: TriNodes,
    *,
    order: int = 2,
    compute_jacfwd: bool = True
) -> Array:
    if compute_jacfwd:
        return value_and_jacfwd(_source_tensor)(x, element, weights, nodes, order=order)
    else:
        return _source_tensor(x, element, weights, nodes, order=order)


@partial(jit, static_argnames=("f", "order"))
def taylor_coeffs(f: Callable[..., Array], element: TriElement, *args: Any, order: int = 2, **kwargs: Any) -> Array:
    n = triangle_normal(element)
    _f = lambda x: f(x, n, *args, **kwargs)
    assert order >= 0

    def charge(c):
        v0 = asarray(_f(c))
        v = v0
        df = _f
        F = [v0[..., None]]
        for i in range(order):
            df = jacfwd(df)
            v = df(c) / math.factorial(i + 1)
            v = v.reshape((*v0.shape, -1))
            F.append(v)

        coefs = jnp.concatenate(F, axis=-1)
        return coefs.swapaxes(-1, 0)

    c = center(element)
    F = jnp.apply_along_axis(charge, -1, c)
    return F


def triangle_normal(element: TriElement) -> NormalVec:
    p0, p1, p2 = element
    v1 = p1 - p0
    v2 = p2 - p0
    n = jnp.cross(v1, v2)
    norm = jnp.linalg.norm(n)

    # Use lax.cond to safely normalize, returning zero if norm is zero
    return lax.cond(
        norm > 0,
        lambda: n / norm,
        lambda: jnp.zeros_like(n),
    )


def single_layer_potential(source_tensor: Sources, charge_tensor: Charges) -> Array:
    return 1 / (4 * pi) * jnp.tensordot(charge_tensor, source_tensor, ((0, 1), (0, 1)))


def curl_single_layer_potential(source_tensor_derivative: Sources, charge_tensor: Charges) -> Array:
    curl_s = -jnp.cross(charge_tensor, source_tensor_derivative)
    return 1 / (4 * pi) * jnp.sum(curl_s, (0, 1))


def scalar_potential_charge(
    mag: Callable[..., Array], phi1: Callable[..., Array], normalized: bool = False
) -> Callable[..., Array]:
    """Creates the SLP charge function for the splitting ansatz of Garcia-Cervera and Roma for the
    scalar potential

    Parameters
    ----------
    mag : Callable[..., Array]
    phi1 : Callable[..., Array]
    normalized : bool, optional
        whether `phi1` is normalized or not, by default False

    Returns
    -------
    Callable[..., Array]
    """

    def charge(x, n, params_mag=(), params_phi1=()):
        if normalized:
            return mag(x, *params_mag) @ n + phi1(x, *params_phi1)
        else:
            return mag(x, *params_mag) @ n - jacfwd(phi1)(x, *params_phi1) @ n

    return charge


def vector_potential_charge(
    mag: Callable[..., Array], A1: Callable[..., Array], normalized: bool = False
) -> Callable[..., Array]:
    """Creates the SLP charge function for the splitting ansatz of Garcia-Cervera and Roma for the
    vector potential

    Parameters
    ----------
    mag : Callable[..., Array]
    A1 : Callable[..., Array]
    normalized : bool, optional
        whether `A1` is normalized or not, by default False

    Returns
    -------
    Callable[..., Array]
    """

    def charge(x, n, params_mag=(), params_A1=()):
        if normalized:
            return cross(mag(x, *params_mag), n) + A1(x, *params_A1)
        else:
            return cross(mag(x, *params_mag), n) - jacfwd(A1)(x, *params_A1) @ n

    return charge


def _source_tensor(
    x: PhySpace, element: TriElement, weights: TriWeights, nodes: TriNodes, *, order: int = 2
) -> Sources:
    assert order >= 0
    c = center(element)

    def _integrand(y, x):
        assert y.shape == (3,)
        assert x.shape == (3,)
        assert c.shape == (3,)
        n = 1 / norm(x - y)
        c1 = asarray([1.0])
        d = y - c
        t = d
        T = [c1]
        for i in range(0, order):
            T.append(t.ravel())
            t = jnp.tensordot(t, d, 0)

        z = jnp.concatenate(T, axis=-1)
        return z * n

    Z = integrate_surface_element(_integrand, element, weights, nodes, x)
    return Z.reshape(-1)
