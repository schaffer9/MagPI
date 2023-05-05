from dataclasses import dataclass
from jax.flatten_util import ravel_pytree

from .prelude import *
from . import calc



class TRResult(T.NamedTuple):
    params: Any
    delta: Array
    grad: Array
    iterations: int
    iterations_steihaug: int


def tr(
    f, params, *args,
    delta0=1., delta_min=1e-4, delta_max=2.,
    eta1=0.15, gamma1=1.2,
    eta2=0.75, gamma2=2.,
    eps_grad=1e-2,
    maxiter=1000,
    maxiter_steihaug=None,
    eps_steihaug=1e-2,
    **kwargs
):
    _f = f
    params, unravel = ravel_pytree(params)
    f = lambda p: _f(unravel(p), *args, **kwargs)
    def step(state):
        _, params, df, delta, i, iter_steihaug = state
        _hvp = lambda p: calc.hvp(f, (params,), (p,))
        _iter_steihaug, p = steihaug(
            df, _hvp, 
            delta=delta, eps=eps_steihaug / (i+1), maxiter=maxiter_steihaug)
        new_params = params + p
        norm_p = norm(p)
        rho = (f(params) - f(new_params)) / -(dot(df, p) + 1 / 2 * dot(p, _hvp(p)))
        
        delta = lax.cond(
            (rho >= eta2) & (jnp.isclose(norm_p, delta)),
            lambda: gamma2 * delta,
            lambda: lax.cond(
                (rho >= eta2),
                lambda: delta,
                lambda: lax.cond(
                    (rho >= eta1) & (jnp.isclose(norm_p, delta)),
                    lambda: gamma1 * delta,
                    lambda: lax.cond(
                        rho >= eta1,
                        lambda: delta,
                        lambda: delta / gamma2
                    )
                )
            )
        )

        params = lax.cond(
            rho >= eta1,
            lambda: new_params,
            lambda: params
        )

        delta = maximum(minimum(delta, delta_max), delta_min)
        df = grad(f)(params)
        break_loop = (norm(df) < eps_grad) | (i + 1 >= maxiter) | (delta <= delta_min) | (norm_p == 0.)
        return (break_loop, params, df, delta, (i + 1), iter_steihaug + _iter_steihaug)

    _, params, df, delta, i, iter_steihaug = lax.while_loop(
        lambda state: ~state[0],
        step,
        (False, params, grad(f)(params), delta0, 0, 0)
    )
    result = TRResult(
        unravel(params),
        delta,
        df,
        i, iter_steihaug
    )
    return result


def steihaug(df, hvp, delta, eps, maxiter=None):
    if maxiter is None:
        if len(df.shape) == 0:
            maxiter = 1
        else:
            maxiter = len(df)
    z = zeros_like(df)
    r = df
    d = -r

    def limit_step(z, d):
        a, b, c = dot(d, d), dot(z, d), dot(z, z) - delta ** 2
        discriminant = b ** 2 - 4 * a * c
        discriminant = lax.cond(discriminant < 0.,lambda: 0., lambda: discriminant)
        tau1 = (-b + sqrt(discriminant)) / (2 * a)
        tau2 = (-b - sqrt(discriminant)) / (2 * a)
        tau = maximum(tau1, tau2)
        z = z + tau * d
        z = z / norm(z) * delta
        return z

    break_loop = False
    i = 0

    def step(state):
        _, i, z, d, r = state
        Hd = hvp(d)
        curvature = dot(d, Hd)
        rTr = dot(r, r)
        alpha = rTr / curvature
        z_new = z + alpha * d
        r_new = r + alpha * Hd
        break_loop, z = lax.cond(
            (curvature <= 0) | (norm(z_new) > delta),
            lambda: (True, limit_step(z, d)),
            lambda: lax.cond(
                norm(r_new) < eps,
                lambda: (True, z_new),
                lambda: (i < maxiter, z_new)
            )
        )
        beta = dot(r_new, r_new) / rTr
        d = -r_new + beta * d
        r = r_new
        return (break_loop, i + 1, z, d, r)

    def condition(state):
        break_loop, *_ = state
        return jnp.invert(break_loop)

    def _steihaug():
        _, iterations, z_new, *_ = lax.while_loop(
            condition,
            step,
            (break_loop, i, z, d, r)
        )
        return iterations, z_new
    
    return lax.cond(norm(r) < eps, lambda: (0, z), _steihaug)
