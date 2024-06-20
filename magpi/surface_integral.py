import math

from .prelude import *
from .integrate import integrate, gauss
from .calc import value_and_jacfwd


def integrate_surface(f, parametrization, u, v, *args, method=gauss(5), **kwargs):
    def integrand(x):
        J = jacfwd(parametrization)(x)
        msg = "Please provide a surface parametrization from [0,1]² -> R³"
        assert J.shape == (3, 2), msg
        ru = J[:, 0]
        rv = J[:, 1]
        z = norm(cross(ru, rv))
        r = parametrization(x)
        return tree_map(lambda y: y * z, f(r, *args, **kwargs))

    return integrate(integrand, [u, v], method=method)


def integrate_surface_elements(f, parametrization, u, v, *args, method=gauss(5), **kwargs):
    def _integrate(u, v):
        c = center(parametrization, u, v)
        return integrate_surface(f, parametrization, u, v, c, *args, method=method, **kwargs)

    assert len(u.shape) == len(v.shape) == 1
    assert u.shape[0] >= 2
    assert v.shape[0] >= 2
    _u = jnp.stack([u[:-1], u[1:]], axis=-1)
    _v = jnp.stack([v[:-1], v[1:]], axis=-1)
    return vmap(vmap(_integrate, (None, 0)), (0, None))(_u, _v)


def center(parametrization, u, v):
    u_bar = (u[..., 1] + u[..., 0]) / 2
    v_bar = (v[..., 1] + v[..., 0]) / 2
    return parametrization(jnp.stack([u_bar, v_bar], axis=-1))


def midpoints(parametrization, u, v):
    def _center(u, v):
        return center(parametrization, u, v)

    _u = jnp.stack([u[:-1], u[1:]], axis=-1)
    _v = jnp.stack([v[:-1], v[1:]], axis=-1)
    return vmap(vmap(_center, (None, 0)), (0, None))(_u, _v)

    
def _source_tensor(x, parametrization, u, v, *, method=gauss(5), order=2):
    assert order >= 0

    def _integrand(y, c, x):
        assert y.shape == (3,)
        assert x.shape == (3,)
        assert c.shape == (3,)

        n = norm(x - y)
        c1 = asarray([1.0])
        d = y - c
        t = d
        T = [c1]
        for i in range(0, order):
            T.append(t.ravel())
            t = jnp.tensordot(t, d, 0)

        c = jnp.concatenate(T, axis=-1)
        return c / n

    Z = integrate_surface_elements(_integrand, parametrization, u, v, x, method=method)
    return Z.reshape(-1)


@partial(jit, static_argnames=("parametrization", "method", "order", "compute_jacfwd"))
def source_tensor(x, parametrization, u, v, *, method=gauss(5), order=2, compute_jacfwd=True):
    if compute_jacfwd:
        return value_and_jacfwd(_source_tensor)(x, parametrization, u, v, method=method, order=order)
    else:
        return _source_tensor(x, parametrization, u, v, method=method, order=order)
    

@partial(jit, static_argnames=("f", "parametrization", "order"))
def charge_tensor(f, parametrization, u, v, *args, order=2, **kwargs):
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

    c = midpoints(parametrization, u, v)
    F = jnp.apply_along_axis(charge, -1, c)
    base_dim = len(c.shape[:-1])
    return F.reshape(-1, *F.shape[base_dim + 1:])


def single_layer_potential(source_tensor, charge_tensor):
    return 1 / (4 * pi) * jnp.tensordot(charge_tensor, source_tensor, ((0,), (0,)))


def curl_single_layer_potential(source_tensor_derivative, charge_tensor):
    curl_s = -jnp.cross(charge_tensor, source_tensor_derivative)
    return 1 / (4 * pi) * jnp.sum(curl_s, 0)


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