import operator
from dataclasses import dataclass

import chex
from jax.experimental import host_callback as hcb
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
    rho: Array
    tr_radius: Array
    aux: Any
    iter_num_steihaug: int
    steihaug_converged: bool | Array
    steihaug_curvature: Array
    steihaug_eps: float | Array


@dataclass(eq=False)
class TR(jaxopt_base.IterativeSolver):
    """
    Trust Region method optimizer which uses the
    CGSteihaug algorithm to solve the trust region sub-problem.

    Attributes
    ----------
    fun: Callable
    value_and_grad: bool
    has_aux: bool = False
    init_tr_radius: float
    max_tr_radius: float
    min_tr_radius: float
    rho_increase: float
    increase_factor: float
    rho_decrease: float
    decrease_factor: float
    rho_accept: float
    tol: float
    maxiter: int
    maxiter_steihaug: int | None
    eps_min_steihaug: float
    """
    fun: Callable
    value_and_grad: bool = False
    has_aux: bool = False
    init_tr_radius: float = 1.0
    max_tr_radius: float = 2.0
    min_tr_radius: float = 1e-10
    rho_increase: float = 3 / 4
    increase_factor: float = 2
    rho_decrease: float = 1 / 4
    decrease_factor: float = 1 / 4
    rho_accept: float = 1 / 4
    tol: float = 1e-2  # gradient tolerance
    maxiter: int = 100
    maxiter_steihaug: int | None = None
    eps_min_steihaug: float = 1e-9

    callback: None | Callable[[jaxopt_base.OptStep], None] = None

    implicit_diff: bool = True
    implicit_diff_solve: Optional[Callable] = None

    jit: jaxopt_base.AutoOrBoolean = "auto"
    unroll: jaxopt_base.AutoOrBoolean = "auto"

    verbose: bool = False

    def init_state(self, init_params: chex.ArrayTree, *args, **kwargs):
        if isinstance(init_params, jaxopt_base.OptStep):
            state = init_params.state
            tr_radius = state.tr_radius
            iter_num = state.iter_num
            init_params = init_params.params
        else:
            iter_num = jnp.asarray(0)
            tr_radius = jnp.asarray(self.init_tr_radius)

        (value, aux), grad = self._value_and_grad_with_aux(init_params, *args, **kwargs)
        norm_df = tree_l2_norm(grad)
        return TrState(
            iter_num=iter_num,
            value=value,
            grad=grad,
            aux=aux,
            error=norm_df,
            rho=jnp.asarray(0.0),
            tr_radius=tr_radius,
            iter_num_steihaug=0,
            steihaug_converged=False,
            steihaug_curvature=jnp.asarray(jnp.inf),
            steihaug_eps=jnp.minimum(1 / 2, norm_df) * norm_df,
        )

    def hvp(
        self, state, params, *args, **kwargs
    ) -> tuple[Any, Callable[[chex.ArrayTree], chex.ArrayTree]]:
        _hvp = lambda p: calc.hvp_forward_over_reverse(
            self._value_and_grad_with_aux,
            (params,),
            (p,),
            *args,
            **kwargs,
            value_and_grad=True,
            has_aux=True
        )
        return state, _hvp

    def update(
        self, params: chex.ArrayTree, state, *args, **kwargs
    ) -> jaxopt_base.OptStep:
        if isinstance(params, jaxopt_base.OptStep):
            params = params.params
        state, hvp = self.hvp(state, params, *args, **kwargs)
        jit, unroll = self._get_loop_options()
        norm_df = state.error
        eps = jnp.minimum(1 / 2, norm_df) * norm_df
        eps = jnp.minimum(state.steihaug_eps, eps)
        steihaug_result = steihaug(
            state.grad,
            hvp,
            tr_radius=state.tr_radius,
            eps=eps,
            maxiter=self.maxiter_steihaug,
            eps_min=self.eps_min_steihaug,
            jit=jit,
            unroll=unroll,
        )
        p = steihaug_result.p

        def update():
            new_params = tree_add(params, p)
            (value, aux), grad = self._value_and_grad_with_aux(
                new_params, *args, **kwargs
            )
            nom = state.value - value
            denom = -(tree_vdot(state.grad, p) + 1 / 2 * tree_vdot(p, hvp(p)))
            rho = nom / denom
            tr_radius = update_tr_radius(
                state.tr_radius,
                self.max_tr_radius,
                self.min_tr_radius,
                rho,
                steihaug_result.step_length,
                self.rho_increase,
                self.rho_decrease,
                self.increase_factor,
                self.decrease_factor,
            )
            accept = rho >= self.rho_accept

            new_state = lax.cond(
                accept,
                lambda: TrState(
                    iter_num=state.iter_num + 1,
                    value=value,
                    grad=grad,
                    error=tree_l2_norm(grad),
                    rho=rho,
                    tr_radius=tr_radius,
                    aux=aux,
                    iter_num_steihaug=steihaug_result.iter_num,
                    steihaug_converged=steihaug_result.converged,
                    steihaug_curvature=steihaug_result.curvature,
                    steihaug_eps=eps
                ),
                lambda: TrState(
                    iter_num=state.iter_num + 1,
                    value=state.value,
                    grad=state.grad,
                    error=state.error,
                    rho=rho,
                    tr_radius=tr_radius,
                    aux=state.aux,
                    iter_num_steihaug=steihaug_result.iter_num,
                    steihaug_converged=steihaug_result.converged,
                    steihaug_curvature=steihaug_result.curvature,
                    steihaug_eps=eps
                ),
            )

            _step = jaxopt.OptStep(
                lax.cond(accept, lambda: new_params, lambda: params), new_state
            )
            if self.callback is not None:
                cb = lambda step, _: self.callback(step)
                hcb.id_tap(cb, _step)

            return _step

        def no_update():
            # in case steihaug does not make any progress, p = 0
            return jaxopt.OptStep(params, state)
        make_update = (steihaug_result.iter_num == 0) | (state.tr_radius <= self.min_tr_radius)
        return lax.cond(make_update, no_update, update)

    def optimality_fun(self, params, *args, **kwargs):
        return self._value_and_grad_with_aux(params, *args, **kwargs)[1]

    def __post_init__(self):
        _, _, self._value_and_grad_with_aux = utils.make_funs_with_aux(
            fun=self.fun, value_and_grad=self.value_and_grad, has_aux=self.has_aux
        )

        if self.rho_accept > self.rho_decrease:
            msg = "`rho_accept` must be smaller than `rho_decrease` for convergence reasons!"
            raise ValueError(msg)


def tree_size(tree: chex.ArrayTree) -> int:
    tree = tree_map(lambda t: t.size, tree)
    return tree_reduce(operator.add, tree)


class CgSteihaugResult(T.NamedTuple):
    iter_num: int
    converged: bool | Array
    limit_step: bool | Array
    step_length: Array
    curvature: Array
    p: chex.ArrayTree


def steihaug(
    grad_f: chex.ArrayTree,
    hvp: Callable[[chex.ArrayTree], chex.ArrayTree],
    tr_radius: float | Array,
    eps: float | Array,
    eps_min: float = 1e-9,
    maxiter: int | None = None,
    unroll: bool = True,
    jit: bool = True,
) -> CgSteihaugResult:  # tuple[Array, int, Array, chex.ArrayTree]:
    if maxiter is None:
        maxiter = tree_size(grad_f)

    z = tree_zeros_like(grad_f)
    r = grad_f
    d = tree_negative(r)

    def limit_step(z, d):
        a, b, c = tree_vdot(d, d), tree_vdot(z, d), tree_vdot(z, z) - tr_radius**2
        discriminant = b**2 - 4 * a * c
        discriminant = lax.cond(discriminant < 0.0, lambda: 0.0, lambda: discriminant)
        tau1 = (-b + sqrt(discriminant)) / (2 * a)
        tau2 = (-b - sqrt(discriminant)) / (2 * a)
        tau = maximum(tau1, tau2)
        z = tree_add_scalar_mul(z, tau, d)
        z = tree_scalar_mul(tr_radius / tree_l2_norm(z), z)
        return z

    def step(state):
        z, d, r = state["z"], state["d"], state["r"]
        rTr = tree_vdot(r, r)
        Hd = hvp(d)
        curvature = tree_vdot(d, Hd)
        alpha = rTr / curvature
        z_new = tree_add_scalar_mul(z, alpha, d)
        r_new = tree_add_scalar_mul(r, alpha, Hd)
        beta = tree_vdot(r_new, r_new) / rTr
        d_new = tree_add_scalar_mul(tree_negative(r_new), beta, d)
        norm_z_new = tree_l2_norm(z_new)
        norm_r_new = tree_l2_norm(r_new)

        z = lax.cond(
            (curvature <= 0) | (norm_z_new >= tr_radius),
            lambda: limit_step(z, d),
            lambda: z_new,
        )

        return dict(
            iter_num=state["iter_num"] + 1,
            z=z,
            d=d_new,
            r=r_new,
            norm_z=norm_z_new,
            norm_r=norm_r_new,
            curvature=curvature,
        )

    def condition(state):
        curvature, norm_z, norm_r = state["curvature"], state["norm_z"], state["norm_r"]
        cond = (
            (curvature <= 0) | (norm_z >= tr_radius) | (norm_r < eps) | (eps < eps_min)
        )
        return jnp.invert(cond)

    dtype = utils.tree_single_dtype(grad_f)
    init_state = dict(
        iter_num=0,
        z=z,
        d=d,
        r=r,
        norm_z=0.0,
        norm_r=tree_l2_norm(r),
        curvature=jnp.asarray(jnp.inf, dtype=dtype),
    )
    state = loop.while_loop(
        condition, step, init_state, maxiter, unroll=unroll, jit=jit
    )
    converged = jnp.invert(condition(state))
    steplength = tree_l2_norm(state["z"])
    result = CgSteihaugResult(
        iter_num=state["iter_num"],
        converged=converged,
        limit_step=jnp.isclose(steplength, tr_radius),
        curvature=state["curvature"],
        step_length=steplength,
        p=state["z"],
    )
    return result


def update_tr_radius(
    tr_radius: Array,
    max_tr_radius: float,
    min_tr_radius: float,
    rho: Array,
    stepsize: Array,
    rho_increase: float,
    rho_decrease: float,
    increase_factor: float,
    decrease_factor: float,
):
    limit_step = jnp.isclose(stepsize, tr_radius)
    tr_radius = lax.cond(
        rho < rho_decrease,
        lambda: tr_radius * decrease_factor,
        lambda: lax.cond(
            (rho > rho_increase) & limit_step,
            lambda: jnp.minimum(tr_radius * increase_factor, max_tr_radius),
            lambda: tr_radius,
        ),
    )
    return jnp.maximum(tr_radius, min_tr_radius)
