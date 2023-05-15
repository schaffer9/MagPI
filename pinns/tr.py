import operator
from dataclasses import dataclass


import chex
from jaxopt import base as jaxopt_base
from jaxopt import loop

from .prelude import *
from . import calc
from . import utils


class TrState(T.NamedTuple):
    iter_num: Array
    value: Array
    grad: Array
    error: Array  # gradient norm
    tr_radius: float
    aux: Any
    iter_num_steihaug: int
    staihaug_converged: bool


@dataclass(eq=False)
class TR(jaxopt_base.IterativeSolver):
    fun: Callable
    value_and_grad: bool = False
    has_aux: bool = False
    init_tr_radius: float = 1.0
    max_tr_radius: float = 2.0
    eta1: float = 3 / 4
    increase_factor: float = 2
    eta2: float = 1 / 4
    decrease_factor: float = 1 / 4
    rho_accept: float = 1 / 6
    tol: float = 1e-2  # gradient tolerance
    maxiter: int = 1000
    maxiter_steihaug: int | None = None
    eps_steihaug: float = 1e-2

    implicit_diff: bool = True
    implicit_diff_solve: Optional[Callable] = None

    jit: jaxopt_base.AutoOrBoolean = "auto"
    unroll: jaxopt_base.AutoOrBoolean = "auto"

    verbose: bool = False

    def init_state(self, init_params: chex.ArrayTree, *args, **kwargs) -> TrState:
        if isinstance(init_params, jaxopt_base.OptStep):
            # `init_params` can either be a pytree or an OptStep object
            state: TrState = init_params.state
            tr_radius = state.tr_radius
            iter_num = state.iter_num
            init_params = init_params.params
        else:
            iter_num = jnp.asarray(0)
            tr_radius = self.init_tr_radius

        (value, aux), grad = self._value_and_grad_with_aux(init_params, *args, **kwargs)
        return TrState(
            iter_num=iter_num,
            value=value,
            grad=grad,
            aux=aux,
            error=jnp.asarray(jnp.inf),
            tr_radius=tr_radius,
            iter_num_steihaug=0,
            staihaug_converged=False,
        )

    def hvp(self, params, *args, **kwargs):
        return lambda p: calc.hvp_forward_over_reverse(
            self._value_and_grad_with_aux,
            (params,),
            (p,),
            *args,
            **kwargs,
            value_and_grad=True,
            has_aux=True
        )

    def update(
        self, params: chex.ArrayTree, state: TrState, *args, **kwargs
    ) -> jaxopt_base.OptStep:
        if isinstance(params, jaxopt_base.OptStep):
            params = params.params
        hvp = self.hvp(params, *args, **kwargs)
        jit, unroll = self._get_loop_options()
        converged, iter_num_steihaug, p = steihaug(
            state.grad,
            hvp,
            tr_radius=state.tr_radius,
            maxiter=self.maxiter_steihaug,
            eps_max=self.eps_steihaug,
            jit=jit,
            unroll=unroll,
        )

        def update():
            new_params = tree_add(params, p)
            (value, aux), grad = self._value_and_grad_with_aux(
                new_params, *args, **kwargs
            )
            nom = state.value - value
            denom = -(tree_vdot(state.grad, p) + 1 / 2 * tree_vdot(p, hvp(p)))
            rho = nom / denom
            norm_p = tree_l2_norm(p)
            tr_radius = update_tr_radius(
                state.tr_radius,
                self.max_tr_radius,
                rho,
                norm_p,
                self.eta1,
                self.eta2,
                self.increase_factor,
                self.decrease_factor,
            )
            accept = rho >= self.rho_accept
            return jaxopt.OptStep(
                lax.cond(accept, lambda: new_params, lambda: params),
                TrState(
                    iter_num=state.iter_num + 1,
                    value=lax.cond(accept, lambda: value, lambda: state.value),
                    grad=lax.cond(accept, lambda: grad, lambda: state.grad),
                    error=lax.cond(
                        accept, lambda: tree_l2_norm(grad), lambda: state.error
                    ),
                    tr_radius=tr_radius,
                    aux=aux,
                    iter_num_steihaug=iter_num_steihaug,
                    staihaug_converged=converged,
                ),
            )

        def no_update():
            # in case steihaug does not make any progress and p = 0
            return jaxopt.OptStep(params, state)

        return lax.cond(iter_num_steihaug == 0, no_update, update)

    def optimality_fun(self, params, *args, **kwargs):
        return self._value_and_grad_with_aux(params, *args, **kwargs)[1]

    def __post_init__(self):
        _, _, self._value_and_grad_with_aux = utils.make_funs_with_aux(
            fun=self.fun, value_and_grad=self.value_and_grad, has_aux=self.has_aux
        )


def tree_elements(tree: chex.ArrayTree) -> int:
    tree = tree_map(lambda t: t.size, tree)
    return tree_reduce(operator.add, tree)


def steihaug(
    grad_f: chex.ArrayTree,
    hvp: Callable[[chex.ArrayTree], chex.ArrayTree],
    tr_radius: float | Array,
    maxiter: int | None,
    eps_max: float = 1e-2,
    unroll: bool = True,
    jit: bool = True,
):
    if maxiter is None:
        maxiter = tree_elements(grad_f)

    z = tree_zeros_like(grad_f)
    r = grad_f
    d = tree_negative(r)
    norm_df = tree_l2_norm(grad_f)

    eps = jnp.minimum(1 / 2, sqrt(norm_df)) * norm_df  # forcing sequence
    eps = jnp.minimum(eps_max, eps)
    eps_min = jnp.finfo(array(1.0).dtype).eps
    eps = jnp.maximum(eps, 10 * eps_min)

    def limit_step(z, d):
        a, b, c = tree_vdot(d, d), tree_vdot(z, d), tree_vdot(z, z) - tr_radius**2
        discriminant = b**2 - 4 * a * c
        discriminant = lax.cond(discriminant < 0.0, lambda: 0.0, lambda: discriminant)
        tau1 = (-b + sqrt(discriminant)) / (2 * a)
        tau2 = (-b - sqrt(discriminant)) / (2 * a)
        tau = maximum(tau1, tau2)
        z = tree_add_scalar_mul(z, tau, d)
        z = tree_scalar_mul(1 / (tree_l2_norm(z) * tr_radius), z)
        return z

    def step(state):
        _, iter_num, z, d, r = state
        Hd = hvp(d)
        curvature = tree_vdot(d, Hd)
        rTr = tree_vdot(r, r)
        alpha = rTr / curvature
        z_new = tree_add_scalar_mul(z, alpha, d)
        r_new = tree_add_scalar_mul(r, alpha, Hd)

        break_loop, z = lax.cond(
            (curvature <= 0) | (tree_l2_norm(z_new) > tr_radius),
            lambda: (True, limit_step(z, d)),
            lambda: (tree_l2_norm(r_new) < eps, z_new),
        )
        beta = tree_vdot(r_new, r_new) / rTr
        d = tree_add_scalar_mul(tree_negative(r_new), beta, d)
        r = r_new
        return (break_loop, iter_num + 1, z, d, r)

    def _steihaug():
        def condition(state):
            break_loop, *_ = state
            return jnp.invert(break_loop)

        converged, iterations, p, *_ = loop.while_loop(
            condition, step, (False, 0, z, d, r), maxiter, unroll=unroll, jit=jit
        )
        return converged, iterations, p

    return lax.cond(norm_df < eps, lambda: (True, 0, z), _steihaug)


def update_tr_radius(
    tr_radius: float,
    max_tr_radius: float,
    rho: Array,
    stepsize: Array,
    eta1: float,
    eta2: float,
    increase_factor: float,
    decrease_factor: float,
):
    limit_step = jnp.isclose(stepsize, tr_radius)
    return lax.cond(
        rho < eta2,
        lambda: tr_radius * decrease_factor,
        lambda: lax.cond(
            (rho > eta1) & limit_step,
            lambda: jnp.minimum(tr_radius * increase_factor, max_tr_radius),
            lambda: tr_radius,
        ),
    )
