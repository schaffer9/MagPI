import operator
from dataclasses import dataclass

import chex
from jax.experimental import io_callback
from jaxopt import base as jaxopt_base
from jax.flatten_util import ravel_pytree
from jaxopt import loop

from .prelude import *
from . import calc
from . import utils


class CgSteihaugResult(T.NamedTuple):
    iter_num: int
    converged: bool | Array
    limit_step: bool | Array
    step_length: Array
    curvature: Array
    p: chex.ArrayTree
    eps: Array
    norm_r: Array


class TrState(T.NamedTuple):
    iter_num: Array
    accepted: bool
    value: Array
    grad: chex.ArrayTree
    error: Array  # gradient norm
    rho: Array
    tr_radius: Array
    aux: Any
    subproblem_result: CgSteihaugResult
    # iter_num_steihaug: int
    # steihaug_converged: bool | Array
    # steihaug_curvature: Array
    # steihaug_eps: float | Array


@dataclass(eq=False)
class TR(jaxopt_base.IterativeSolver):
    """
    Trust Region method optimizer which uses the
    CGSteihaug algorithm to solve the trust region sub-problem.
    """
    fun: Callable
    value_and_grad: bool = False
    has_aux: bool = False
    init_tr_radius: float = 1.0
    max_tr_radius: float = 2.0
    min_tr_radius: float = 1e-4
    rho_increase: float = 3 / 4
    increase_factor: float = 2
    rho_decrease: float = 1 / 4
    decrease_factor: float = 1 / 4
    rho_accept: float = 1 / 4
    forcing_parameter: float = 1 / 2
    damping_factor: float | None = None
    tol: float = 1e-2  # gradient tolerance
    maxiter: int = 100
    maxiter_steihaug: int | None = None
    approx_hvp: bool = False
    fd_h: float = 1e-4

    callback: None | Callable[[jaxopt_base.OptStep], None] = None

    implicit_diff: bool = True
    implicit_diff_solve: Callable | None = None

    jit: bool = True
    unroll: jaxopt_base.AutoOrBoolean = "auto"

    verbose: bool = False

    def init_state(self, init_params: chex.ArrayTree, *args, **kwargs):
        if isinstance(init_params, jaxopt_base.OptStep):
            state = init_params.state
            tr_radius = state.tr_radius
            tr_radius = lax.cond(
                tr_radius <= self.min_tr_radius,
                lambda: self.init_tr_radius,
                lambda: tr_radius
            )
            iter_num = state.iter_num
            init_params = init_params.params
        else:
            iter_num = jnp.asarray(0)
            tr_radius = jnp.asarray(self.init_tr_radius)
        
        (value, aux), grad_f = self._value_and_grad_with_aux(init_params, *args, **kwargs)
        norm_df = tree_l2_norm(grad_f)
        subproblem_result = CgSteihaugResult(
            0, False, False, asarray(0.0), asarray(0.0), tree_zeros_like(init_params),
            asarray(jnp.inf), asarray(jnp.inf),
        )
        return TrState(
            iter_num=iter_num,
            accepted=False,
            value=value,
            grad=grad_f,
            error=norm_df,
            aux=aux,
            rho=jnp.asarray(0.0),
            tr_radius=tr_radius,
            subproblem_result=subproblem_result,
        )

    def update(
        self, params: chex.ArrayTree, state, *args, **kwargs
    ) -> jaxopt_base.OptStep:
        if isinstance(params, jaxopt_base.OptStep):
            params = params.params

        if self.approx_hvp:
            def fd_hvp(p, old_grad):
                _params = tree_add(
                    params, tree_scalar_mul(self.fd_h, p)
                )
                _, df = self._value_and_grad_with_aux(_params, *args, **kwargs)
                Hp = tree_scalar_mul(1 / self.fd_h, tree_sub(df, old_grad))
                return Hp
            
            (old_value, old_aux), old_grad = self._value_and_grad_with_aux(
                params, *args, **kwargs
            )
            hvp = lambda p: fd_hvp(p, old_grad)
        else:
            old_value, old_grad, hvp, old_aux = calc.value_grad_hvp(
                self._value_and_grad_with_aux, params, *args,
                value_and_grad=True, has_aux=True, **kwargs
            )
        if self.damping_factor is None:
            _hvp = hvp
        else:
            _hvp = lambda x: tree_add_scalar_mul(hvp(x), self.damping_factor, x)
        
        unroll = self._get_unroll_option()
        norm_df = tree_l2_norm(old_grad)
        eps = jnp.minimum(1 / 2, norm_df ** self.forcing_parameter) * norm_df
        eps = jnp.minimum(state.subproblem_result.eps, eps)
        eps = jnp.maximum(self.tol, eps)
        steihaug_result = steihaug(
            old_grad,
            _hvp,
            tr_radius=state.tr_radius,
            eps=eps,
            maxiter=self.maxiter_steihaug,
            jit=self.jit,
            unroll=unroll,
        )
        p = steihaug_result.p

        def update():
            new_params = tree_add(params, p)
  
            (value, aux), new_grad = self._value_and_grad_with_aux(
                new_params, *args, **kwargs
            )
            nom = old_value - value
            denom = -(tree_vdot(old_grad, p) + 1 / 2 * tree_vdot(p, hvp(p)))
            rho = (nom) / (denom)
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
                    accepted=True,
                    value=value,
                    grad=new_grad,
                    error=tree_l2_norm(new_grad),
                    rho=rho,
                    tr_radius=tr_radius,
                    aux=aux,
                    subproblem_result=steihaug_result,
                ),
                lambda: TrState(
                    iter_num=state.iter_num + 1,
                    accepted=False,
                    value=old_value,
                    grad=old_grad,
                    error=tree_l2_norm(old_grad),
                    rho=rho,
                    tr_radius=tr_radius,
                    aux=old_aux,
                    subproblem_result=steihaug_result
                ),
            )

            _step = jaxopt.OptStep(
                lax.cond(
                    accept,
                    lambda: new_params,
                    lambda: params
                ),
                new_state
            )
            if self.callback is not None:
                io_callback(self.callback, None, _step)

            return _step

        def no_update():
            # in case steihaug does not make any progress, p = 0
            return jaxopt.OptStep(params, state)
        make_update = (state.tr_radius <= self.min_tr_radius)
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
        
        super().__post_init__()


def tree_size(pytree: chex.ArrayTree) -> int:
    pytree = tree_map(lambda t: t.size, pytree)
    return tree_reduce(operator.add, pytree)


def steihaug(
    grad_f: chex.ArrayTree,
    hvp: Callable[[chex.ArrayTree], chex.ArrayTree],
    tr_radius: float | Array,
    eps: float | Array,
    maxiter: int | None = None,
    unroll: bool = True,
    jit: bool = True,
) -> CgSteihaugResult:  # tuple[Array, int, Array, chex.ArrayTree]:
    if maxiter is None:
        maxiter = 10 * tree_size(grad_f)

    z = tree_zeros_like(grad_f)
    r = grad_f
    d = tree_negative(r)

    def limit_step(z, d):
        a, b, c = tree_vdot(d, d), tree_vdot(z, d), tree_vdot(z, z) - tr_radius**2
        discriminant = (b / a) ** 2 - c / a
        discriminant = lax.cond(discriminant < 0.0, lambda: 0.0, lambda: discriminant)
        tau = - b / a + sqrt(discriminant)
        tau = jnp.maximum(0.0, tau)
        z = tree_add_scalar_mul(z, tau, d)
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
        cond = (curvature <= 0) | (norm_z >= tr_radius) | (norm_r < eps)
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
        norm_r=state["norm_r"],
        eps=asarray(eps),
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
