import math

from .prelude import *
from .integrate import integrate, gauss, integrate_quad_rule
from .calc import value_and_jacfwd



def from_ref_element(x, vertices):
    v0, v1, v2 = vertices
    y = (1 - x[..., 0] - x[..., 1]) * v0 + x[..., 0] * v1 + x[..., 1] * v2
    return y


def integrate_surface_element(f, vertices, weights, nodes, *args, **kwargs):
    parametrization = lambda x: from_ref_element(x, vertices)
    def integrand(x):
        r, J = value_and_jacfwd(parametrization)(x)
        msg = "Please provide a surface parametrization from [0,1]² -> R³"
        assert J.shape == (3, 2), msg
        ru = J[:, 0]
        rv = J[:, 1]
        z = norm(cross(ru, rv))
        return tree_map(lambda y: y * z, f(r, *args, **kwargs))

    return integrate_quad_rule(integrand, weights, nodes)


def center(vertices):
    return from_ref_element(array([1 / 3, 1 / 3]), vertices)


def _source_tensor(x, vertices, weights, nodes, *, order=2):
    assert order >= 0
    c = center(vertices)

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

    Z = integrate_surface_element(_integrand, vertices, weights, nodes, x)
    return Z.reshape(-1)


@partial(jit, static_argnames=("order", "compute_jacfwd"))
def source_tensor(x, vertices, weights, nodes, *, order=2, compute_jacfwd=True):
    if compute_jacfwd:
        return value_and_jacfwd(_source_tensor)(x, vertices, weights, nodes, order=order)
    else:
        return _source_tensor(x, vertices, weights, nodes, order=order)
    

@partial(jit, static_argnames=("f", "order"))
def charge_tensor(f, vertices, *args, order=2, **kwargs):
    _f = lambda x: f(x, *args, **kwargs)
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

    c = center(vertices)
    F = jnp.apply_along_axis(charge, -1, c)
    #base_dim = len(c.shape[:-1])
    #return F.reshape(-1, *F.shape[base_dim + 1:])
    return F


def single_layer_potential(source_tensor, charge_tensor):
    return 1 / (4 * pi) * jnp.tensordot(charge_tensor, source_tensor, ((0, 1), (0, 1)))


def curl_single_layer_potential(source_tensor_derivative, charge_tensor):
    curl_s = -jnp.cross(charge_tensor, source_tensor_derivative)
    return 1 / (4 * pi) * jnp.sum(curl_s, (0, 1))


def scalar_potential_charge(adf, mag, phi1, normalized=False):
    if normalized:
        def charge(x, params_mag=(), params_phi1=()):
            n = -jacfwd(adf)(x)
            n = n / norm(n)  # it is still important to divide by norm(n) for AD
            return mag(x, *params_mag) @ n + phi1(x, *params_phi1)
        
        return charge
    else:
        def charge(x, params_mag=(), params_phi1=()):
            n = -jacfwd(adf)(x)
            n = n / norm(n)
            return mag(x, *params_mag) @ n - jacfwd(phi1)(x, *params_phi1) @ n
        
        return charge
        
    
def vector_potential_charge(adf, mag, A1, normalized=False):
    if normalized:
        def charge(x, params_mag=(), params_A1=()):
            n = -jacfwd(adf)(x)
            n = n / norm(n)
            return cross(mag(x, *params_mag), n) + A1(x, *params_A1)
        
        return charge
    else:
        def charge(x, params_mag=(), params_A1=()):
            n = -jacfwd(adf)(x)
            n = n / norm(n)
            return cross(mag(x, *params_mag), n) - jacfwd(A1)(x, *params_A1) @ n
        
        return charge
    

def source_tensor_for_mesh(x, mesh, weights, nodes, *, order=2, compute_jacfwd=True):
    def source_for_element(i):
        vertices = mesh.nodes[i]
        z = source_tensor(x, vertices, weights, nodes, order=order, compute_jacfwd=compute_jacfwd)
        return z

    if compute_jacfwd:
        z, dz = vmap(source_for_element)(mesh.elements)
        return z, dz
    else:
        z = vmap(source_for_element)(mesh.elements)
        z = z.reshape(-1)
        return z
    

def charge_tensor_for_mesh(f, mesh, *args, order=2, **kwargs):
    def inner(i):
        vertices = mesh.nodes[i]
        s = charge_tensor(f, vertices, *args, order=order, **kwargs)
        return s
    s = vmap(inner)(mesh.elements)
    return s
