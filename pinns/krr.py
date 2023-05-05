from flax.struct import dataclass, field
from jaxopt.linear_solve import solve_cg

from .prelude import *

__all__ = (
    "krr", "KRR", "rbf"
)


def rbf(x, y, gamma=1.):
    d = jnp.sum((x - y) ** 2, axis=-1)
    return exp(-gamma * d)


def pairwise_kernel(k, x, y):
    return vmap(vmap(k, (None, 0)), (0, None))(x, y)


@dataclass
class KRR:
    coef: Array
    support: Array
    kernel: Callable = field(compare=False)

    def __call__(self, x):
        K = vmap(self.kernel, (None, 0))(x, self.support)
        return K @ self.coef


def krr(kernel, x, y, alpha, **solver_kwargs) -> KRR:
    K = pairwise_kernel(kernel, x, x)
    coef = solve_cg(lambda x: K @ x, y, ridge=alpha, **solver_kwargs)
    return KRR(coef, x, kernel)
