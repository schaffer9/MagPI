import math
from dataclasses import dataclass

import chex
from jax.experimental import host_callback as hcb
from jaxopt import base as jaxopt_base
from jaxopt import loop
from jax.scipy.optimize import minimize

from .prelude import *
from . import calc
from . import utils


class SFNState(T.NamedTuple):
    iter_num: chex.ArrayTree
    value: Array
    error: Array
    aux: Any
    tr_radius: Array
    rho: Array
    grad: chex.ArrayTree
    last_update: chex.ArrayTree
    # subproblem_converged: bool | Array
    # subproblem_iter_num: int
    # lbfgs_status: int
    # lbfgs_value: Array
    # lbfgs_iter: int


@dataclass(eq=False)
class SFN(jaxopt_base.IterativeSolver):
    fun: Callable
    value_and_grad: bool = False
    has_aux: bool = False
    k: int = 10
    init_tr_radius: float = 1.0
    max_tr_radius: float = 2.0
    min_tr_radius: float = 1e-10
    rho_increase: float = 3 / 4
    increase_factor: float = 2
    rho_decrease: float = 1 / 4
    decrease_factor: float = 1 / 4
    rho_accept: float = 1 / 4
    damping_parameter: float | None = None
    maxiter: int = 100
    maxiter_subproblem: int | None = None
    tol: float = 1e-2  # gradient tolerance
    tol_subproblem: float = 1e-5
    bfgs_tol: float = 1e-5
    bfgs_options: dict[str, Any] | None = None
    callback: None | Callable[[jaxopt_base.OptStep], None] = None

    implicit_diff: bool = True
    implicit_diff_solve: Optional[Callable] = None

    jit: bool = True
    unroll: jaxopt_base.AutoOrBoolean = "auto"

    verbose: bool = False

    def init_state(self, init_params: chex.ArrayTree, *args, **kwargs):
        if isinstance(init_params, jaxopt_base.OptStep):
            state = init_params.state
            iter_num = state.iter_num
            init_params = init_params.params
            tr_radius = state.tr_radius
        else:
            iter_num = asarray(0)
            tr_radius = jnp.asarray(self.init_tr_radius)
        (value, aux), grad_f = self._value_and_grad_with_aux(init_params, *args, **kwargs)
        norm_df = tree_l2_norm(grad_f)
        return SFNState(
            iter_num=iter_num,
            value=value,
            error=norm_df,
            aux=aux,
            tr_radius=tr_radius,
            rho=jnp.asarray(0.0),
            grad=grad_f,
            last_update=tree_zeros_like(init_params),
            # subproblem_converged=False,
            # subproblem_iter_num=0,
            # lbfgs_status=-1,
            # lbfgs_value=asarray(jnp.inf),
            # lbfgs_iter=0
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
        self, params: chex.ArrayTree, state: SFNState, *args, **kwargs
    ) -> jaxopt_base.OptStep:
        if isinstance(params, jaxopt_base.OptStep):
            state = params.state
            params = params.params
            
        state, hvp = self.hvp(state, params, *args, **kwargs)
        unroll = self._get_unroll_option()
        
        # norm_df = state.error
        # eps = jnp.minimum(self.forcing_parameter, norm_df ** self.forcing_parameter) * norm_df
        
        V, W, _, _ = lanczos_iteration(self.k, tree_negative(state.grad), hvp)#, state.last_update)
        G = tree_matmul(V, W)
        eigval, eigvec = jnp.linalg.eigh(G)

        if self.damping_parameter is None:
            H  = eigvec.T * (eigval) @ eigvec
            H_inv = eigvec.T * 1 / jnp.abs(eigval) @ eigvec
        else:
            H = eigvec.T * (eigval + self.damping_parameter) @ eigvec
            H_inv = eigvec.T * 1 / (jnp.abs(eigval) + self.damping_parameter) @ eigvec
        
        g = tree_matvec(V, state.grad)
        H_inv_g = H_inv @ g
        lag_multiplier = -jnp.sqrt((g @ H_inv_g) / (2 * state.tr_radius))
        alpha = H_inv_g / lag_multiplier
        params_update = tree_map(lambda v: jnp.tensordot(alpha, v, ((0,), (0,))), V)
        params_new = tree_add(params, params_update)

        (value_f, aux), grad_f = self._value_and_grad_with_aux(
            params_new, *args, **kwargs
        )
        nom = state.value - value_f
        # denom = -g @ alpha - 1 / 2 * alpha @ H @ alpha
        denom = -tree_vdot(state.grad, params_update) - 1 / 2 * tree_vdot(params_update, hvp(params_update))
        rho = nom / denom
        tr_radius = update_tr_radius(
            state.tr_radius,
            self.max_tr_radius,
            self.min_tr_radius,
            rho,
            self.rho_increase,
            self.rho_decrease,
            self.increase_factor,
            self.decrease_factor,
        )
        accept = rho >= self.rho_accept

        new_state = lax.cond(
            accept,
            lambda: SFNState(
                iter_num=state.iter_num + asarray(1),
                value=value_f,
                error=tree_l2_norm(grad_f),
                rho=rho,
                tr_radius=tr_radius,
                grad=grad_f,
                aux=aux,
                last_update=tree_sub(params_new, params),
            ),
            lambda: SFNState(
                iter_num=state.iter_num + asarray(1),
                value=state.value,
                error=state.error,
                tr_radius=tr_radius,
                rho=rho,
                aux=state.aux,
                grad=state.grad,
                last_update=state.last_update
            ),
        )

        _step = jaxopt.OptStep(
            lax.cond(accept, lambda: params_new, lambda: params), new_state
        )
        if self.callback is not None:
            cb = lambda step, _: self.callback(step)
            hcb.id_tap(cb, _step)

        return _step

        # state = SFNState(
        #     iter_num=state.iter_num + asarray(1),
        #     value=_state["value"],
        #     error=tree_l2_norm(_state["grad"]),
        #     aux=_state["aux"],
        #     tr_radius=state.tr_radius,
        #     rho=rho,
        #     grad=_state["grad"],
        #     last_update=params_update,
        # )
        # _step = jaxopt_base.OptStep(params_new, state)
        # if self.callback is not None:
        #     cb = lambda step, _: self.callback(step)
        #     hcb.id_tap(cb, (_step, eigval, V, W, _state))
        # return _step


        # if self.damping_parameter is None:
        #     H_inv = lambda lam: eigvec.T * 1 / (jnp.abs(eigval) + lam) @ eigvec
        # else:
        #     H_inv = lambda lam: eigvec.T * 1 / (jnp.abs(eigval) + self.damping_parameter + lam) @ eigvec
        
        # def alpha_dot_V(alpha):
        #     return tree_map(lambda v: jnp.tensordot(alpha, v, ((0,), (0,))), V)
          
        # def fun(lam, params, g):
        #     lam = lam[0]
        #     alpha = H_inv(lam) @ g
        #     p = tree_add(params, alpha_dot_V(alpha))
        #     value, _ = self._fun(p, *args, **kwargs)
        #     return value
        
        # def body_fun(state):
        #     params_old = state["params_new"]
        #     (value_f, aux), grad_f = self._value_and_grad_with_aux(params_old, *args, **kwargs)
        #     g = -tree_matvec(V, grad_f)
            
            
        #     # todo: minimize!!
        #     result = minimize(fun, asarray([state["lam"]]), (params_old, g),
        #                       tol=self.bfgs_tol, method="BFGS",
        #                       options=self.bfgs_options)  
            
            
        #     lam = result.x[0]
        #     alpha = H_inv(lam) @ g
        #     params_update = alpha_dot_V(alpha)
        #     params_new = tree_add(params_old, params_update)
        #     error = tree_l2_norm(tree_sub(params_old, params_new))
        #     return dict(
        #         iter_num=state["iter_num"] + 1, lam=lam,
        #         params_new=params_new, params_old=params_old, error=error,
        #         value=value_f, aux=aux, grad=grad_f, converged=error < self.tol_subproblem,
        #         lbfgs_status=result.status, lbfgs_value=result.fun, lbfgs_iter=result.nit
        #     )
        
        # _init_state = dict(
        #     iter_num=0, lam=0.0,
        #     params_new=params, params_old=params, error=jnp.inf,
        #     value=state.value, aux=state.aux, grad=state.grad,
        #     converged=False, lbfgs_status=-1, lbfgs_value=state.value, lbfgs_iter=0,
        # )
        
        # def cond_fun(state):
        #     return jnp.invert(state["converged"])
        
        # if self.maxiter_subproblem is None:
        #     _maxiter = dim(params)
        # else:
        #     _maxiter = self.maxiter_subproblem
        # _state = loop.while_loop(cond_fun, body_fun, _init_state,
        #                          maxiter=_maxiter, unroll=unroll, jit=self.jit)
        # converged = jnp.invert(cond_fun(_state))
        # params_new = _state["params_old"]
        # params_update = tree_sub(params_new, params)
        # state = SFNState(
        #     iter_num=state.iter_num + asarray(1),
        #     value=_state["value"],
        #     grad=_state["grad"],
        #     aux=_state["aux"],
        #     last_update=params_update,
        #     error=tree_l2_norm(_state["grad"]),
        #     subproblem_converged=converged,
        #     subproblem_iter_num=_state["iter_num"],
        #     lbfgs_status=_state["lbfgs_status"],
        #     lbfgs_value=_state["lbfgs_value"],
        #     lbfgs_iter=_state["lbfgs_iter"]
        # )
        # _step = jaxopt_base.OptStep(params_new, state)
        # if self.callback is not None:
        #     cb = lambda step, _: self.callback(step)
        #     hcb.id_tap(cb, (_step, eigval, V, W, _state))
        # return _step

    def optimality_fun(self, params, *args, **kwargs):
        return self._value_and_grad_with_aux(params, *args, **kwargs)[1]

    def __post_init__(self):
        self._fun, _, self._value_and_grad_with_aux = utils.make_funs_with_aux(
            fun=self.fun, value_and_grad=self.value_and_grad, has_aux=self.has_aux
        )

        super().__post_init__()


def lanczos_iteration(k, w0, matvec_product, last_update=None, unroll=False, jit=True):
    alpha = zeros((k,))
    beta = zeros((k,))
    W = tree_map(lambda t: zeros((k, *t.shape)), w0)
    V = tree_map(lambda t: zeros((k, *t.shape)), w0)
    w_last = w0
    v_last = tree_zeros_like(w0)
    
    def body(i, state):
        (V, W, alpha, beta, v_last, w_last) = state
        beta_i = tree_l2_norm(w_last)
        vi = tree_scalar_mul(1 / beta_i, w_last)
        if last_update is None:
            wi = matvec_product(vi)
        else:
            wi = lax.cond(
                tree_is_zero(last_update),
                lambda: matvec_product(vi),
                lambda: lax.cond(i == k - 1, lambda: last_update, lambda: matvec_product(vi))
            )
        W = _tree_set(W, wi, i)
        V = _tree_set(V, vi, i)
        
        alpha_i = tree_vdot(wi, vi)
        wi = tree_sub(wi, tree_scalar_mul(alpha_i, vi))
        wi = tree_sub(wi, tree_scalar_mul(beta_i, v_last))
        alpha = alpha.at[i].set(alpha_i)
        beta = beta.at[i].set(beta_i)
        return (V, W, alpha, beta, vi, wi)

    state = (V, W, alpha, beta, v_last, w_last)
    (V, W, alpha, beta, _, _) = _for_loop(0, k, body, state, unroll=unroll, jit=jit)
    return V, W, alpha, beta


def _tree_set(M, v, i):
    return tree_map(lambda M, v: M.at[i].set(v), M, v)


def tree_is_zero(t):
    return tree_reduce(lambda a, b: a & jnp.all(b == 0), t, jnp.array(True))


def _for_loop(lower, upper, body_fun, init_val, unroll=False, jit=True):
    def cond_fun(state):
        i, _ = state
        return i < upper
        
    def body(state):
        i, state = state
        state = body_fun(i, state)
        return i + 1, state
    
    _, state = loop.while_loop(cond_fun, body, (lower, init_val), 
                               maxiter=upper, unroll=unroll, jit=jit)
    return state


def tree_matmul(a, b):
    vmap_left = jax.vmap(tree_vdot, in_axes=(0, None))
    vmap_right = jax.vmap(vmap_left, in_axes=(None, 0))
    return vmap_right(a, b)


def tree_matvec(a, b):
    _matmul = jax.vmap(tree_vdot, in_axes=(0, None))
    return _matmul(a, b)


def dim(params):
    return sum([math.prod(p.shape) for p in tree_leaves(params)])


def update_tr_radius(
    tr_radius: Array,
    max_tr_radius: float,
    min_tr_radius: float,
    rho: Array,
    rho_increase: float,
    rho_decrease: float,
    increase_factor: float,
    decrease_factor: float,
):
    tr_radius = lax.cond(
        rho < rho_decrease,
        lambda: tr_radius * decrease_factor,
        lambda: lax.cond(
            (rho > rho_increase),
            lambda: jnp.minimum(tr_radius * increase_factor, max_tr_radius),
            lambda: tr_radius,
        ),
    )
    return jnp.maximum(tr_radius, min_tr_radius)
