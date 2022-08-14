from jaxopt.linear_solve import solve_normal_cg
from dataclasses import dataclass, field

from .prelude import *


@dataclass(frozen=True)
class ELM:
    coef: ndarray
    W: ndarray
    b: ndarray
    activation: Callable = field(compare=False)

    def __call__(self, x):
        return self.activation(x @ self.W + self.b) @ self.coef
        

def elm(X, y, W, b, activation=tanh, init=None, **solver_kwargs):
    slp = vmap(lambda x: activation(W @ x + b))
    H = slp(X)
    if init is None and len(y.shape) == 1:
        init = tree_map(lambda W: zeros((W.shape[0],)), W)
    elif init is None and len(y.shape) > 1:
        init = tree_map(lambda W, y: zeros((W.shape[0], y.shape[-1])), W, y)
    p = solve_normal_cg(lambda x: H @ x, y, init=init, **solver_kwargs)
    return ELM(p, jnp.transpose(W), b, activation)