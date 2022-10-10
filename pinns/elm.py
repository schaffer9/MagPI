from jaxopt.linear_solve import solve_normal_cg
from flax.struct import dataclass, field
from .prelude import *

Array = Any
# Single hidden layer feedforward network
SLFN = Callable[[Array], Array]


@dataclass
class ELM:
    coef: Array
    slfn: Callable[[Array], Array] = field(compare=False)

    def __call__(self, x):
        return self.slfn(x) @ self.coef


# def elm_from_weights(X: Array, y: Array, W: Array, b: Array, 
#                      activation: Optional[Callable[[Array], Array]] = None, 
#                      **solver_kwargs) -> ELM:
#     if activation is None:
#         activation = tanh
#     slfn = lambda x: activation(W @ x + b)
#     return elm(slfn, X, y, **solver_kwargs)


def elm(slfn: SLFN, X: Array, y: Array, **solver_kwargs) -> ELM:
    H = vmap(slfn)(X)
    if "init" in solver_kwargs.keys():
        init = init
    else:
        if len(y.shape) > 1:
            init = tree_map(lambda H, y: zeros((H.shape[-1], y.shape[-1])), H, y)
        else:
            init = tree_map(lambda H: zeros((H.shape[-1],)), H)
    params = solve_normal_cg(lambda x: H @ x, y, init=init, **solver_kwargs)
    return ELM(params, slfn)
