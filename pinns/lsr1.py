from dataclasses import dataclass

import chex
from jaxopt import base as jaxopt_base
from jaxopt import loop

from .prelude import *


class Lsr1State(T.NamedTuple):
    iter_num: int
    value: float
    grad: chex.ArrayTree
    tr_radius: float
    error: float | Array
    aux: Any | None = None
    

@dataclass(eq=False)
class LSR1(jaxopt_base.IterativeSolver):
    fun: Callable[..., Array]
    value_and_grad: bool = False
    has_aux: bool = False

    maxiter: int = 100
    tol: float = 1e-4
    history_size: int = 10
    init_tr_radius: float = 1.0

    implicit_diff: bool = True
    implicit_diff_solve: Optional[Callable] = None

    jit: jaxopt_base.AutoOrBoolean = "auto"
    unroll: jaxopt_base.AutoOrBoolean = "auto"

    verbose: bool = False

    def init_state(
        self, init_params: chex.ArrayTree, key: random.PRNGKeyArray, *args, **kwargs
    ):
        if isinstance(init_params, jaxopt_base.OptStep):
            # `init_params` can either be a pytree or an OptStep object
            state_kwargs = dict(
                
            )
            init_params = init_params.params
            dtype = tree_single_dtype(init_params)
        else:
            dtype = tree_single_dtype(init_params)
            state_kwargs = dict(
                
            )
        (value, aux), grad = self._value_and_grad_with_aux(
            init_params, self.dataset, *args, **kwargs
        )
        return Lsr1State(
            value=value,
            grad=grad,
            aux=aux,
            error=jnp.asarray(jnp.inf),
            **state_kwargs,
        )


    def update(
        self, params: chex.ArrayTree, state: Lsr1State, key, *args, **kwargs
    ) -> jaxopt_base.OptStep:
        if isinstance(params, jaxopt_base.OptStep):
            params = params.params
        pass

    def optimality_fun(self, params, *args, **kwargs):
        return self._value_and_grad_fun(params, *args, **kwargs)[1]

    def _value_and_grad_fun(self, params, *args, **kwargs):
        if isinstance(params, jaxopt_base.OptStep):
            params = params.params
        (value, aux), grad = self._value_and_grad_with_aux(params, *args, **kwargs)
        return value, grad

    def __post_init__(self):
        pass


def sample_params(params, memory_size, radius):
    pass

def adjust_trust_region(tr_radius, reduction_ratio):
    pass


def hvp(f, primals, tangents, *args, value_and_grad=False, has_aux=False, **kwargs):
    def grad_f(p):
        if value_and_grad:
            _, _grad_f = f(p, *args, **kwargs)
        else:
            _, _grad_f = jax.value_and_grad(f, has_aux=has_aux)(p, *args, **kwargs)
        return _grad_f

    return jvp(grad_f, primals, tangents)[1]



def tree_single_dtype(tree):
    """The dtype for all values in e tree."""
    dtypes = set(p.dtype for p in tree_leaves(tree) if isinstance(p, Array))
    if not dtypes:
        return None
    if len(dtypes) == 1:
        return dtypes.pop()
    raise ValueError("Found more than one dtype in the tree.")
