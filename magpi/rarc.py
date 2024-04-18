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
    eigenvalues: Eigenvalues
    eigenvectors: Eigenvectors


class SubproblemResult(T.NamedTuple):
    iter_num: int
    p: Array
    error: Array
    eigenvalues: Eigenvalues
    converged: Array


class RARCState(T.NamedTuple):
    iter_num: chex.ArrayTree
    key: Key
    value: Array
    error: Array
    aux: Any
    alpha: Array
    rho: Array
    accepted: Array
    grad: chex.ArrayTree
    last_update: chex.ArrayTree
    eig: EighResult
    search_space: Array
    search_curvature: Array
    reduced_update: Array
    subproblem_result: SubproblemResult


@dataclass(eq=False)
class RARC(jaxopt_base.IterativeSolver):
    fun: Callable
    value_and_grad: bool = False
    has_aux: bool = False
    r: int = 10
    q: int = 3
    key: Array = field(default_factory=lambda: random.PRNGKey(0))
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
    tol_subproblem: float = 1e-3
    eig_select_threshold: float = 0.0
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
        else:
            iter_num = asarray(0)
            alpha = jnp.asarray(self.init_alpha)
        (value, aux), grad_f = self._value_and_grad_with_aux(
            init_params, *args, **kwargs
        )
        norm_df = tree_l2_norm(grad_f)
        subproblem_res = SubproblemResult(
            0,
            zeros((self.r,)),
            jnp.asarray(jnp.inf),
            zeros((self.r,)),
            jnp.asarray(False),
        )
        key, subkey2 = random.split(self.key)
        grad_f_ravel, _ = ravel_pytree(grad_f)
        # U = random.normal(subkey1, (*grad_f_ravel.shape, self.r))
        # U, _ = jnp.linalg.qr(U)
        V = random.normal(subkey2, (*grad_f_ravel.shape, self.r))

        return RARCState(
            iter_num=iter_num,
            key=key,
            value=value,
            error=norm_df,
            aux=aux,
            alpha=alpha,
            rho=jnp.asarray(0.0),
            accepted=jnp.asarray(False),
            grad=grad_f,
            last_update=tree_zeros_like(init_params),
            eig=EighResult(zeros((self.r,)), zeros((self.r, self.r))),
            search_space=V,
            search_curvature=zeros((self.r,)),
            reduced_update=zeros((self.r,)),
            subproblem_result=subproblem_res,
        )

    def update(
        self, params: chex.ArrayTree, state: RARCState, *args, **kwargs
    ) -> jaxopt_base.OptStep:
        if isinstance(params, jaxopt_base.OptStep):
            state = params.state
            params = params.params

        value, grad_f, hvp, aux = calc.value_grad_hvp(
            self._value_and_grad_with_aux,
            params,
            *args,
            has_aux=True,
            value_and_grad=True,
            **kwargs
        )
        unroll = self._get_unroll_option()

        _g, unravel = ravel_pytree(grad_f)

        def _hvp(Q):
            Q_new = hvp(unravel(Q))
            Q_new, _ = ravel_pytree(Q_new)
            return Q_new

        key, subkey = random.split(state.key)
        # eig = approx_eigh(state.eig.eigenvectors, _hvp, self.q, unroll=unroll)
        # P = sample_search_space(subkey, eig, threshold=self.eig_select_threshold)
        P = sample_search_space(subkey, state)

        # compute reduced hessian and reduced eigendecomposition
        H = P.T @ vmap(_hvp, -1, -1)(P)
        curvature = diag(H)
        S, U = jnp.linalg.eigh(H)
        if self.damping_parameter is not None:
            S = asarray(jnp.where(jnp.abs(S) < self.damping_parameter, self.damping_parameter, S))

        # diagonalize system
        g = U.T @ P.T @ _g
        if self.maxiter_subproblem is None:
            maxiter_subproblem = dim(params)
        else:
            maxiter_subproblem = self.maxiter_subproblem

        result = solve_subproblem(
            state.alpha,
            g,
            S,
            maxiter_subproblem,
            self.tol_subproblem,
            unroll=unroll,
            jit=self.jit,
        )

        # update params and regularization parameter
        p = result.p
        params_update = unravel(P @ U @ result.p)
        params_new = tree_add(params, params_update)
        (value_new, aux_new), grad_f_new = self._value_and_grad_with_aux(
            params_new, *args, **kwargs
        )
        machine_eps = jnp.finfo(S.dtype).eps
        nom = (value - value_new) + machine_eps
        denom = (
            -(g @ p + 1 / 2 * (p * S) @ p + state.alpha / 3 * norm(p) ** 3)
            + machine_eps
        )
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
                accepted=jnp.asarray(True),
                alpha=alpha,
                grad=grad_f_new,
                aux=aux_new,
                last_update=params_update,
                #eig=eig,
                eig=EighResult(S, U),
                search_space=P,
                #search_curvature=S,
                search_curvature=curvature,
                reduced_update=p,
                subproblem_result=result,
            ),
            lambda: RARCState(
                iter_num=state.iter_num + asarray(1),
                key=key,
                value=value,
                error=tree_l2_norm(grad_f),
                alpha=alpha,
                rho=rho,
                accepted=jnp.asarray(False),
                aux=aux,
                grad=grad_f,
                last_update=state.last_update,
                # eig=eig,
                eig=EighResult(S, U),
                search_space=P,
                #search_curvature=S,
                search_curvature=curvature,
                reduced_update=p,
                subproblem_result=result,
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


# def sample_search_space(key, eig: EighResult, threshold):
#     U, S = eig.eigenvectors, eig.eigenvalues
#     V = random.normal(key, eig.eigenvectors.shape)
#     V = V - (U @ (U.T @ V))
#     V, _ = jnp.linalg.qr(V)
#     V = jnp.where((S < threshold)[None, :], U, V)
#     print(V.shape)
#     return V
#     #V = jnp.concatenate([U, V], axis=-1)
#     #return V

def sample_search_space(key, state: RARCState):
    V_new = random.normal(key, state.search_space.shape)
    V_old = state.search_space
    V = jnp.where((jnp.abs(state.reduced_update) > state.search_curvature)[None, :], V_old, V_new)
    V, _ = jnp.linalg.qr(V)
    return V


# def approx_eigh(
#     Q: Array,
#     matvec: Callable[[Array], Array],
#     q: int,
#     unroll: None | int | bool = None
# ) -> EighResult:
#     """Performs an Eigenvalue Decomposition via random subspace iteration.
#     See Algorithm 4.4 [1].

#     Parameters
#     ----------
#     Q : chex.ArrayTree
#         initial Eigenvector guess
#     matvec : Callable[[Array], Array]
#         Hermitian matrix vector product
#     q : int
#         number of iterations
#     unroll : None | int | bool, optional
#         unroll option for `jax.lax.fori_loop`

#     Returns
#     -------
#     EighResult

#     Notes
#     -----
#     .. [1] Halko, Nathan, Per-Gunnar Martinsson, and Joel A. Tropp.
#     "Finding structure with randomness: Probabilistic algorithms for
#     constructing approximate matrix decompositions."
#     SIAM review 53.2 (2011): 217-288.
#     """
#     # def body(i, Q_lam):
#     #     Q, _ = Q_lam
#     #     Q = vmap(matvec, -1, -1)(Q)
#     #     Q, R = jnp.linalg.qr(Q)
#     #     return Q, diag(R)
    
#     # Q, lam = lax.fori_loop(0, q, body, (Q, zeros((Q.shape[-1]),)), unroll=unroll)

#     def body(i, Q):
#         Q = vmap(matvec, -1, -1)(Q)
#         Q, _ = jnp.linalg.qr(Q)
#         return Q
    
#     Q = lax.fori_loop(0, q, body, Q, unroll=unroll)

#     def compute_lam(q):
#         return q @ matvec(q)
    
#     lam = vmap(compute_lam, -1)(Q)

#     return EighResult(lam, Q)


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
    alpha: float | Array : regularization parameter
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

    def solution_eigenvalues(p):
        # eigenvalues with rank-one update are bounded from below,
        # see: Golub, Gene H. "Some modified matrix eigenvalue problems."
        # SIAM review 15.2 (1973): 318-334.
        return S + alpha * norm(p)

    p0 = -grad_f
    iter_num, p = loop.while_loop(
        cond, body, (0, p0), maxiter=maxiter, unroll=unroll, jit=jit
    )
    error = norm(dp(p))
    S_sol = solution_eigenvalues(p)
    return SubproblemResult(iter_num, p, error, S_sol, (norm(dp(p)) <= tol))
