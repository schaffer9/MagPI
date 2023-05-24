from dataclasses import dataclass

import chex
from jax import tree_util
from jaxopt import base as jaxopt_base

from pinns.krr import pairwise_kernel
import operator
from jax.experimental import host_callback as hcb

from jaxopt import implicit_diff as idf
from jaxopt.base import OptStep
from jaxopt._src.base import Solver
from pinns.tr import TrState
from jax import tree_util
import functools

from .prelude import *
from .tr import TR, TrState, update_tr_radius, steihaug
from . import utils
from jax.flatten_util import ravel_pytree
from jaxopt import loop


class RandomIterativeSolver(Solver):
    """Base class for iterative solvers.

    Any iterative solver should implement `init` and `update` methods:
      - `state = init_state(init_params, *args, **kwargs)`
      - `next_params, next_state = update(params, state, *args, **kwargs)`

    This class implements a `run` method:

      `params, state = run(init_params, *args, **kwargs)`

    The following attributes are needed by the `run` method:
      - `verbose`
      - `maxiter`
      - `tol`
      - `implicit_diff`
      - `implicit_diff_solve`

    If `implicit_diff` is not present, it is assumed to be True.

    The following attribute is needed in the state:
      - `error`
    """

    def _get_loop_options(self):
        """Returns jit and unroll options based on user-provided attributes."""

        if self.jit == "auto":
            # We always jit unless verbose mode is enabled.
            jit = not self.verbose
        else:
            jit = self.jit

        if self.unroll == "auto":
            # We unroll when implicit diff is disabled or when jit is disabled.
            unroll = not getattr(self, "implicit_diff", True) or not jit
        else:
            unroll = self.unroll

        return jit, unroll

    def _cond_fun(self, inputs):
        _, state = inputs[1]
        if self.verbose:
            print("error:", state.error)
        return state.error > self.tol

    def _body_fun(self, inputs):
        key, (params, state), (args, kwargs) = inputs
        key, update_key = random.split(key)
        return (
            key,
            self.update(update_key, params, state, *args, **kwargs),
            (args, kwargs),
        )

    # TODO(frostig,mblondel): temporary workaround to accommodate line
    # search as an iterative solver, but for this reason and others
    # (automatic implicit diff) we should consider having it not be one.
    def _make_zero_step(self, init_params, state) -> OptStep:
        if isinstance(init_params, OptStep):
            return OptStep(params=init_params.params, state=state)
        else:
            return OptStep(params=init_params, state=state)

    def _run(self, key, init_params: Any, *args, **kwargs) -> OptStep:
        key, init_key, update_key = random.split(key, 3)
        state = self.init_state(init_key, init_params, *args, **kwargs)

        # We unroll the very first iteration. This allows `init_val` and `body_fun`
        # below to have the same output type, which is a requirement of
        # lax.while_loop and lax.scan.
        #
        # TODO(frostig,mblondel): if we could check concreteness of self.maxiter,
        # and we knew that it is concrete here, then we could optimize away the
        # redundant first step, e.g.:
        #
        #   maxiter = get_maybe_concrete(self.maxiter)  # concrete value or None
        #   if maxiter == 0:
        #     return OptStep(params=init_params, state=state)
        #
        # In the general case below, we prefer to use `jnp.where` instead
        # of a `lax.cond` for now in order to avoid staging the initial
        # update and the run loop. They might not be staging compatible.

        zero_step = self._make_zero_step(init_params, state)

        opt_step = self.update(update_key, init_params, state, *args, **kwargs)
        init_val = (key, opt_step, (args, kwargs))

        jit, unroll = self._get_loop_options()

        many_step = loop.while_loop(
            cond_fun=self._cond_fun,
            body_fun=self._body_fun,
            init_val=init_val,
            maxiter=self.maxiter - 1,
            jit=jit,
            unroll=unroll,
        )[1]

        return tree_util.tree_map(
            functools.partial(_where, self.maxiter == 0),
            zero_step,
            many_step,
            is_leaf=lambda x: x is None,
        )  # state attributes can sometimes be None

    def run(
        self, key: random.PRNGKeyArray, init_params: Any, *args, **kwargs
    ) -> OptStep:
        """Runs the optimization loop.

        Args:
          init_params: pytree containing the initial parameters.
          *args: additional positional arguments to be passed to the update method.
          **kwargs: additional keyword arguments to be passed to the update method.
        Returns:
          (params, state)
        """
        run = self._run

        if getattr(self, "implicit_diff", True):
            reference_signature = getattr(self, "reference_signature", None)
            decorator = idf.custom_root(
                self.optimality_fun,
                has_aux=True,
                solve=self.implicit_diff_solve,
                reference_signature=reference_signature,
            )
            run = decorator(run)

        return run(key, init_params, *args, **kwargs)


def _where(cond, x, y):
    if x is None:
        return y
    if y is None:
        return x
    return jnp.where(cond, x, y)


# class Lsr1State(T.NamedTuple):
#     key: random.PRNGKeyArray
#     iter_num: Array
#     value: Array
#     grad: Array
#     error: Array  # gradient norm
#     rho: Array
#     tr_radius: Array
#     aux: Any
#     iter_num_steihaug: int
#     steihaug_converged: bool | Array
#     steihaug_curvature: Array

@dataclass(eq=False)
class Lsr1(RandomIterativeSolver, TR):
    memory_size: int = 10

    def hvp(self, key, state: Any, params, *args, **kwargs):
        S = sample_s(key, params, self.memory_size)
        state, _hvp = super().hvp(state, params, *args, **kwargs)
        Y = vmap(_hvp, 
                 axis_name="hvp_axis",
                 axis_size=self.memory_size,
                 spmd_axis_name="hvp_axis")(S)
        return state, ls1_hvp(S, Y)

    def init_state(
        self, key, init_params: chex.ArrayTree, *args, **kwargs
    ) -> TrState:
        return super().init_state(init_params, *args, **kwargs)
        # return Lsr1State(key, *tr_state)

    def update(
        self, key, params: chex.ArrayTree, state, *args, **kwargs
    ) -> jaxopt_base.OptStep:
        if isinstance(params, jaxopt_base.OptStep):
            params = params.params
        state, hvp = self.hvp(key, state, params, *args, **kwargs)
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


def sample_s(
    key: random.PRNGKeyArray, params: chex.ArrayTree, m: int
) -> chex.ArrayTree:
    dtype = utils.tree_single_dtype(params)
    _params = tree_util.tree_leaves(params)
    keys = random.split(key, len(_params))
    treedef = tree_util.tree_structure(params)
    keys = tree_util.tree_unflatten(treedef, keys)
    rand_params = tree_map(
        lambda p, k: random.uniform(k, (m,) + p.shape, dtype, -1, 1), params, keys
    )
    return tree_negative(rand_params)
    #norms = vmap(lambda p: tree_l2_norm(p))(rand_params)
    #rand_params = tree_vector_mul(rand_params, 1 / norms)
    #rand_params = tree_add(params, tree_scalar_mul(r, rand_params))
    #rand_params = vmap(lambda p, n: tree_div(p, n))(rand_params, norms)
    #return tree_sub(params, rand_params)


def tree_vector_mul(tree, v):
    return tree_map(lambda p: p * v[:, *([None] * (len(p.shape) - 1))], tree)


kernel = lambda a, b: tree_vdot(a, b)
import numbers

def ls1_hvp(S, Y, B0=None):
    if B0 is None:
        s0 = tree_map(lambda s: s[0], S)
        B0 = tree_ones_like(s0)

    elif isinstance(B0, numbers.Number) or B0.shape == ():
        # B0 is scalar
        s0 = tree_map(lambda s: s[0], S)
        B0 = tree_scalar_mul(B0, tree_ones_like(s0))

    B0_v = lambda v: tree_map(operator.mul, v, B0)
    STY = pairwise_kernel(kernel, S, Y)
    BS = vmap(B0_v)(S)
    STBS = pairwise_kernel(kernel, S, BS)
    K = STY + STBS
    _U, singular_values, _VT = jnp.linalg.svd(K, hermitian=True)
    #singular_values = singular_values + 1e-6
    #singular_values = jnp.where(jnp.abs(singular_values) > 1e-6, singular_values, 1e-6)
    K_inv = _VT.T * (1 / singular_values) @ _U.T
    YsubBS = tree_sub(Y, BS)

    def hvp(p):
        h = vmap(kernel, (0, None))(YsubBS, p)
        r = K_inv @ h
        e = tree_map(lambda A: jnp.tensordot(r, A, axes=1), YsubBS)
        return tree_add(B0_v(p), e)

    return hvp


# res = ls1_hvp(S, Y, 1.)(p)
# tree_map(jnp.shape, res)
