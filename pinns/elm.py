from jaxopt.linear_solve import solve_normal_cg
from flax.struct import dataclass, field
from .prelude import *

# Single hidden layer feedforward network
SLFN = Callable[[Array], Array]


@jit
def solve_svd(A, b):
    u, s, vh = jax.scipy.linalg.svd(A, full_matrices=False, lapack_driver="gesvd")# jnp.linalg.svd(A, full_matrices=False)
    u = u * (1 / s)
    return vh.T @ (u.T @ b)


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
        init = solver_kwargs["init"]
    else:
        if len(y.shape) > 1:
            init = tree_map(lambda H, y: zeros((H.shape[-1], y.shape[-1])), H, y)
        else:
            init = tree_map(lambda H: zeros((H.shape[-1],)), H)
    # params = solve_normal_cg(lambda x: H @ x, y, init=init, **solver_kwargs)
    params = solve_svd(H, y)
    return ELM(params, slfn)
