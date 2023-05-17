import chex
from jax import tree_util
from jaxopt import base as jaxopt_base

from jaxopt import implicit_diff as idf
from pinns.tr import TrState

import functools

from .prelude import *
from .tr import TR, TrState
from . import utils
from jax.flatten_util import ravel_pytree
from jaxopt import loop


class _RandomIterativeSolverMixin:
    def _run(
        self, key: random.PRNGKeyArray, init_params: Any, *args, **kwargs
    ) -> jaxopt_base.OptStep:
        state = self.init_state(key, init_params, *args, **kwargs)
        zero_step = self._make_zero_step(init_params, state)
        opt_step = self.update(init_params, state, *args, **kwargs)
        init_val = (opt_step, (args, kwargs))
        jit, unroll = self._get_loop_options()
        many_step = loop.while_loop(
            cond_fun=self._cond_fun,
            body_fun=self._body_fun,
            init_val=init_val,
            maxiter=self.maxiter - 1,
            jit=jit,
            unroll=unroll,
        )[0]

        return tree_util.tree_map(
            functools.partial(_where, self.maxiter == 0),
            zero_step,
            many_step,
            is_leaf=lambda x: x is None,
        )  # state attributes can sometimes be None

    def run(
        self, key: random.PRNGKeyArray, init_params: Any, *args, **kwargs
    ) -> jaxopt_base.OptStep:
        return super().run(key, init_params, *args, **kwargs)
        # run = self._run

        # if getattr(self, "implicit_diff", True):
        #     reference_signature = getattr(self, "reference_signature", None)
        #     decorator = idf.custom_root(
        #         self.optimality_fun,
        #         has_aux=True,
        #         solve=self.implicit_diff_solve,
        #         reference_signature=reference_signature,
        #     )
        #     run = decorator(run)

        # return run(key, init_params, *args, **kwargs)


def _where(cond, x, y):
    if x is None:
        return y
    if y is None:
        return x
    return jnp.where(cond, x, y)


class Lsr1State(T.NamedTuple):
    key: random.PRNGKeyArray
    iter_num: Array
    value: Array
    grad: Array
    error: Array  # gradient norm
    tr_radius: float
    aux: Any
    iter_num_steihaug: int
    staihaug_converged: bool


class Lsr1(_RandomIterativeSolverMixin, TR):
    memory_size: int = 10

    def hvp(self, state: Lsr1State, params, *args, **kwargs):
        S = sample_s(state.key, params, self.memory_size)
        _hvp = super().hvp(params, *args, **kwargs)
        Y = _hvp(S)

        # compute inverse matrix
        ...

        # def _hvp(v):
        #     pass

        # return _hvp

    def init_state(
        self, key, init_params: chex.ArrayTree, *args, **kwargs
    ) -> Lsr1State:
        tr_state = super().init_state(init_params, *args, **kwargs)
        return Lsr1State(self.key, *tr_state)

    def update(
        self, params: chex.ArrayTree, state: Lsr1State, *args, **kwargs
    ) -> jaxopt_base.OptStep:
        params, tr_state = super().update(params, state, *args, **kwargs)
        return jaxopt_base.OptStep(
            params, Lsr1State(random.split(state.key)[0], *tr_state)
        )


from jax import tree_util
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
    norms = vmap(lambda p: tree_l2_norm(p))(rand_params)
    rand_params = tree_vector_mul(rand_params, 1 / norms)
    return tree_sub(params, rand_params)


def tree_vector_mul(tree, v):
    return tree_map(lambda p: p * v[:, *([None] * (len(p.shape) - 1))], tree)


from pinns.krr import pairwise_kernel
import operator
kernel = lambda a, b: tree_vdot(a, b)

def tree_dim(tree: chex.ArrayTree, axis=0) -> int:
    dim = tree_leaves(tree_map(lambda t: t.shape[axis], tree))
    dim = set(dim)
    assert len(dim) == 1, f"Dimension mismatch! All arrays must have same size along axis {axis}."
    return dim.pop()

def tree_single_dtype(tree):
    """The dtype for all values in e tree."""
    dtypes = set(p.dtype for p in tree_leaves(tree) if isinstance(p, Array))
    if not dtypes:
        return None
    if len(dtypes) == 1:
        return dtypes.pop()
    raise ValueError("Found more than one dtype in the tree.")


def ls1_hvp(S, Y, B0=None):

    if B0 is None:
        s0 = tree_map(lambda s: s[0], S)
        B0 = tree_ones_like(s0)
    
    if not isinstance(B0, Array) or B0.shape == ():
        # B0 is scalar
        s0 = tree_map(lambda s: s[0], S)
        B0 = tree_scalar_mul(B0, tree_ones_like(s0))
        
    B0_v = lambda v: tree_map(operator.mul, v, B0)
    STY = pairwise_kernel(kernel, S, Y)
    BS = vmap(B0_v)(S)
    STBS = pairwise_kernel(kernel, S, BS)
    K = STY + STBS
    _U, singular_values, _VT= jnp.linalg.svd(K, hermitian=True)
    K_inv = (_VT.T * (1 / singular_values) @ _U.T)
    YsubBS = tree_sub(Y, BS)
    def hvp(p):
        h = vmap(kernel, (0, None))(YsubBS, p)
        r = K_inv @ h
        e = tree_map(lambda A: jnp.tensordot(r, A, axes=1), YsubBS)
        return tree_add(B0_v(p), e)
        
    return hvp


# res = ls1_hvp(S, Y, 1.)(p)
# tree_map(jnp.shape, res)