from jaxopt.linear_solve import solve_normal_cg
from dataclasses import dataclass, field
from .prelude import *

Array = ndarray

@dataclass(frozen=True)
class ELM:
    coef: Array
    W: Array
    b: Array
    activation: Callable[[Array], Array] = field(compare=False)

    def __call__(self, x):
        return self.activation(x @ self.W + self.b) @ self.coef
        

def elm(X: Array, y: Array, W: Array, b: Array, 
        activation: Optional[Callable[[Array], Array]]=None, 
        init: Optional[Array]=None, **solver_kwargs) -> ELM:
    if activation is None:
        activation = tanh
    slp = vmap(lambda x: activation(W @ x + b))
    H = slp(X)
    if init is None and len(y.shape) == 1:
        init = tree_map(lambda W: zeros((W.shape[0],)), W)
    elif init is None and len(y.shape) > 1:
        init = tree_map(lambda W, y: zeros((W.shape[0], y.shape[-1])), W, y)
    p = solve_normal_cg(lambda x: H @ x, y, init=init, **solver_kwargs)
    return ELM(p, jnp.transpose(W), b, activation)