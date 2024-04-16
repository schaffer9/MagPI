import math
from dataclasses import dataclass, field

import chex
from jax.experimental import host_callback as hcb
from jaxopt import base as jaxopt_base
from jaxopt import loop
from jax.flatten_util import ravel_pytree

from .prelude import *
from . import calc
from . import utils

Key = Array
Eigenvalues = Array
Eigenvectors = Array


class EighResult(T.NamedTuple):
    eigenvalues: jax.Array
    eigenvectors: jax.Array


class SubproblemResult(T.NamedTuple):
    iter_num: int
    p: Array
    error: Array
    converged: Array


class RARCState(T.NamedTuple):
    iter_num: chex.ArrayTree
    key: Array
    value: Array
    error: Array
    aux: Any
    alpha: Array
    rho: Array
    grad: chex.ArrayTree
    last_update: chex.ArrayTree
    subproblem_result: SubproblemResult


@dataclass(eq=False)
class RARC(jaxopt_base.IterativeSolver):
    fun: Callable
    value_and_grad: bool = False
    has_aux: bool = False
    r: int = 10
    q: int = 2
    init_key: Array = field(default_factory=lambda: random.PRNGKey(0))
    init_alpha: float = 1.0
    alpha_min: float = 1e-5
    rho_success: float = 3 / 4
    decrease_factor: float = 1 / 2
    rho_failure: float = 1 / 4
    increase_factor: float = 4.0
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
            alpha = state.tr_radius
            init_key = state.key
        else:
            iter_num = asarray(0)
            alpha = jnp.asarray(self.init_alpha)
            init_key = self.init_key
        (value, aux), grad_f = self._value_and_grad_with_aux(
            init_params, *args, **kwargs
        )
        norm_df = tree_l2_norm(grad_f)
        subproblem_res = SubproblemResult(0, zeros((self.r,)),
                                          jnp.asarray(jnp.inf),
                                          jnp.asarray(False))
        return RARCState(
            iter_num=iter_num,
            key=init_key,
            value=value,
            error=norm_df,
            aux=aux,
            alpha=alpha,
            rho=jnp.asarray(0.0),
            grad=grad_f,
            last_update=tree_zeros_like(init_params),
            subproblem_result=subproblem_res
        )

    def update(
        self, params: chex.ArrayTree, state: RARCState, *args, **kwargs
    ) -> jaxopt_base.OptStep:
        if isinstance(params, jaxopt_base.OptStep):
            state = params.state
            params = params.params

        value, grad_f, hvp, aux = calc.value_grad_hvp(
            self._value_and_grad_with_aux, params, *args,
            has_aux=True, value_and_grad=True, **kwargs
        )
        unroll = self._get_unroll_option()
        key, subkey = random.split(state.key)
        S, U = random_eigh(subkey, hvp, grad_f, self.r, self.q, unroll=unroll)
        
        if self.damping_parameter is not None:
            S = jnp.where(jnp.abs(S) < self.damping_parameter, self.damping_parameter, S)

        g = tree_matvec(U, grad_f)
        
        if self.maxiter_subproblem is None:
            maxiter_subproblem = dim(params)
        else:
            maxiter_subproblem = self.maxiter_subproblem
            
        result = solve_subproblem(state.alpha, g, S, maxiter_subproblem,
                                  self.tol_subproblem, unroll=unroll, jit=self.jit)

        p = result.p
        params_update = tree_map(lambda v: v @ p, U)
        params_new = tree_add(params, params_update)
        (value_new, aux_new), grad_f_new = self._value_and_grad_with_aux(
            params_new, *args, **kwargs
        )
        eps = jnp.finfo(S.dtype).eps
        nom = (value - value_new) + eps
        denom = -(g @ p + 1 / 2 * (p * S) @ p + state.alpha / 3 * norm(p) ** 3) + eps
        rho = nom / denom
        alpha = update_regularization_parameter(
            state.alpha,
            self.alpha_min,
            rho,
            self.rho_success,
            self.rho_failure,
            self.decrease_factor,
            self.increase_factor,
        )
        
        accept = rho >= self.rho_accept

        new_state = lax.cond(
            accept,
            lambda: RARCState(
                iter_num=state.iter_num + asarray(1),
                key=key,
                value=value_new,
                error=tree_l2_norm(grad_f_new),
                rho=rho,
                alpha=alpha,
                grad=grad_f_new,
                aux=aux_new,
                last_update=params_update,
                subproblem_result=result
            ),
            lambda: RARCState(
                iter_num=state.iter_num + asarray(1),
                key=key,
                value=value,
                error=tree_l2_norm(grad_f),
                alpha=alpha,
                rho=rho,
                aux=aux,
                grad=grad_f,
                last_update=state.last_update,
                subproblem_result=result
            ),
        )
        
        _step = jaxopt.OptStep(
            lax.cond(accept, lambda: params_new, lambda: params), new_state
        )
        if self.callback is not None:
            cb = lambda step, _: self.callback(step)
            hcb.id_tap(cb, _step)

        return _step

    def optimality_fun(self, params, *args, **kwargs):
        return self._value_and_grad_with_aux(params, *args, **kwargs)[1]

    def __post_init__(self):
        self._fun, _, self._value_and_grad_with_aux = utils.make_funs_with_aux(
            fun=self.fun, value_and_grad=self.value_and_grad, has_aux=self.has_aux
        )  
        super().__post_init__()


def tree_matvec(a, b):
    _matmul = jax.vmap(tree_vdot, in_axes=(-1, None), out_axes=-1)
    return _matmul(a, b)


def dim(params) -> int:
    return int(sum([math.prod(p.shape) for p in tree_leaves(params)]))


def update_regularization_parameter(
    alpha: Array,
    min_alpha: float,
    rho: Array,
    rho_success: float,
    rho_failure: float,
    decrease_factor: float,
    increase_factor: float,
):
    alpha = lax.cond(
        rho > rho_success,
        lambda: jnp.maximum(alpha * decrease_factor, min_alpha),
        lambda: lax.cond(
            (rho > rho_failure),
            lambda: alpha,
            lambda: alpha * increase_factor,
        ),
    )
    return alpha


def random_eigh(
    key: Key,
    matvec: Callable[[chex.ArrayTree], chex.ArrayTree],
    Q_init: chex.ArrayTree,
    r: int,
    q: int = 2,
    unroll: None | int | bool = None,
) -> EighResult:
    """Performs an Eigenvalue Decomposition via random subspace iteration.
    See Algorithm 4.4 and 5.3 in [1].

    Parameters
    ----------
    key : Key
    matvec : Callable[[Array], Array]
        Hermitian matrix vector product
    Q_init : chex.ArrayTree
        initial structure of vector applied to `matvec`
    r : int
        computes the `r` largest Eigenpairs of `matvec`
    q : int
        number of power iterations to run, by default 2
    unroll : None | int | bool, optional
        unroll option for `jax.lax.fori_loop`

    Returns
    -------
    tuple[Eigenvalues, Eigenvectors]

    Notes
    -----
    .. [1] Halko, Nathan, Per-Gunnar Martinsson, and Joel A. Tropp.
    "Finding structure with randomness: Probabilistic algorithms for
    constructing approximate matrix decompositions."
    SIAM review 53.2 (2011): 217-288.
    """
    Q, unravel = ravel_pytree(Q_init)
    Q = random.normal(key, (*Q.shape, r))

    # subspace iteration:

    def _matvec(Q):
        Q = matvec(unravel(Q))
        Q, _ = ravel_pytree(Q)
        return Q

    def body(i, Q):
        Q = vmap(_matvec, -1, -1)(Q)
        Q, _ = jnp.linalg.qr(Q)
        return Q

    Q = lax.fori_loop(1, q, body, Q, unroll=unroll)
    # Eigenvalue decomposition:
    Y = vmap(_matvec, -1, -1)(Q)
    B = Q.T @ Y
    S, U = jnp.linalg.eigh(B)
    U = Q @ U
    U = vmap(unravel, -1, -1)(U)
    return EighResult(S, U)


def solve_subproblem(
    alpha: float | Array,
    grad_f: Array,
    S: Eigenvalues,
    maxiter: int,
    tol: float,
    unroll: bool,
    jit: bool,
) -> SubproblemResult:
    """Newton iteration for the Cubic Regularization subproblem with diagonal Hessian
    and Sherman–Morrison formula.

    Parameters
    ----------
    grad_f : Array
        gradient of f times the Eigenvectors
    S : Eigenvalues
        Eigenvalues of the Hessian
    maxiter : int
    tol : float
        stopping tolerance for the gradient of the subproblem
    unroll : bool
    jit : bool
    """
    def body(state):
        iter_num, p = state
        grad_p = grad_f + S * p + alpha * norm(p) * p
        H_inv = 1 / (S + alpha * norm(p))
        q = p * jnp.sqrt(alpha / norm(p))
        Hq = H_inv * q
        p = p - (H_inv * grad_p - Hq * (Hq @ grad_p) / (1 + q @ Hq))
        return iter_num + 1, p

    def dp(p):
        return grad_f + S * p + alpha * norm(p) * p
    
    def cond(state):
        return norm(dp(state[1])) > tol
    
    def _pinv():
        lam0 = S[0]
        _S = S + lam0
        S_pinv = _S.at[1:].set(1 / _S[1:])
        return -S_pinv * grad_f
    
    p0 = lax.cond(
        S[0] > tol,
        lambda: -(1 / S) * grad_f,
        _pinv,
    )
    iter_num, p = loop.while_loop(cond, body, (0, p0), maxiter=maxiter, unroll=unroll, jit=jit)
    error = norm(dp(p))
    return SubproblemResult(iter_num, p, error, norm(dp(p)) <= tol)
