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
    grad: chex.ArrayTree
    error: Array
    last_update: chex.ArrayTree
    aux: Any
    subproblem_converged: bool | Array
    subproblem_iter_num: int
    lbfgs_status: int
    lbfgs_value: Array
    lbfgs_iter: int


@dataclass(eq=False)
class SFN(jaxopt_base.IterativeSolver):
    fun: Callable
    value_and_grad: bool = False
    has_aux: bool = False
    k: int = 10
    damping_parameter: float | None = None
    # forcing_parameter: float = 1 / 2
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
        else:
            iter_num = 0
        (value, aux), grad = self._value_and_grad_with_aux(init_params, *args, **kwargs)
        norm_df = tree_l2_norm(grad)
        return SFNState(
            iter_num=asarray(iter_num),
            value=value,
            grad=grad,
            aux=aux,
            error=norm_df,
            last_update=tree_zeros_like(init_params),
            subproblem_converged=False,
            subproblem_iter_num=0,
            lbfgs_status=-1,
            lbfgs_value=asarray(jnp.inf),
            lbfgs_iter=0
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
        
        V, W, _, _ = lanczos_iteration(self.k, state.grad, hvp, state.last_update)
        G = tree_matmul(V, W)
        eigval, eigvec = jnp.linalg.eigh(G)
        if self.dampling_parameter is None:
            H_inv = lambda lam: eigvec.T * 1 / (jnp.abs(eigval) + lam) @ eigvec
        else:
            H_inv = lambda lam: eigvec.T * 1 / (jnp.abs(eigval) + self.dampling_parameter + lam) @ eigvec
            
        def fun(lam, params, g):
            lam = lam[0]
            alpha = H_inv(lam) @ g
            p = tree_add(params, tree_map(lambda v: alpha @ v, V))
            value, _ = self._fun(p, *args, **kwargs)
            return value
        
        def body_fun(state):
            params_old = state["params_new"]
            (value_f, aux), grad_f = self._value_and_grad_with_aux(params_old, *args, **kwargs)
            g = -tree_matvec(V, grad_f)
            result = minimize(fun, array([1.0]), (params_old, g),
                              tol=self.bfgs_tol, method="BFGS",
                              options=self.bfgs_options)
            lam = result.x[0]
            alpha = H_inv(lam) @ g
            params_update = tree_map(lambda v: alpha @ v, V)
            params_new = tree_add(params_old, params_update)
            error = tree_l2_norm(tree_sub(params_old, params_new))
            return dict(
                iter_num=state["iter_num"] + 1,
                params_new=params_new, params_old=params_old, error=error,
                value=value_f, aux=aux, grad=grad_f, converged=error < self.tol_subproblem,
                lbfgs_status=result.status, lbfgs_value=result.fun, lbfgs_iter=result.nit
            )
        
        _init_state = dict(
            iter_num=0,
            params_new=params, params_old=params, error=jnp.inf,
            value=state.value, aux=state.aux, grad=state.grad,
            converged=False, lbfgs_status=-1, lbfgs_value=state.value, lbfgs_iter=0,
        )
        
        def cond_fun(state):
            return jnp.invert(state["converged"])
        
        if self.maxiter_subproblem is None:
            _maxiter = dim(params)
        else:
            _maxiter = self.maxiter_subproblem
        _state = loop.while_loop(cond_fun, body_fun, _init_state,
                                 maxiter=_maxiter, unroll=unroll, jit=self.jit)
        converged = jnp.invert(cond_fun(_state))
        params_new = _state["params_old"]
        params_update = tree_sub(params_new, params)
        state = SFNState(
            iter_num=state.iter_num + asarray(1),
            value=_state["value"],
            grad=_state["grad"],
            aux=_state["aux"],
            last_update=params_update,
            error=tree_l2_norm(_state["grad"]),
            subproblem_converged=converged,
            subproblem_iter_num=_state["iter_num"],
            lbfgs_status=_state["lbfgs_status"],
            lbfgs_value=_state["lbfgs_value"],
            lbfgs_iter=_state["lbfgs_iter"]
        )
        _step = jaxopt_base.OptStep(params_new, state)
        if self.callback is not None:
            cb = lambda step, _: self.callback(_step)
            hcb.id_tap(cb, _step)
        return _step

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