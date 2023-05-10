from dataclasses import dataclass

import chex
from jaxopt import base as jaxopt_base
from jaxopt import loop

from .prelude import *


def dataset_size(dataset):
    n = tree_map(lambda d: d.shape[0], dataset)
    n = tree_leaves(n)
    assert all([_n == n[0] for _n in n])
    return n[0]


class SlbfgsState(T.NamedTuple):
    key: random.PRNGKeyArray
    iter_num: int
    value: float
    grad: chex.ArrayTree
    stepsize: float
    error: float | Array
    s_history: chex.ArrayTree
    y_history: chex.ArrayTree
    rho_history: Array
    param_history: chex.ArrayTree
    gamma: Array
    aux: Any | None = None
    last: int = 0
    last_param: int = 0


@dataclass(eq=False)
class SLBFGS(jaxopt_base.IterativeSolver):
    fun: Callable[..., Array]
    dataset: chex.ArrayTree
    value_and_grad: bool = False
    has_aux: bool = False

    maxiter: int = 100
    tol: float = 1e-4
    history_size: int = 10
    batch_size_gradient: int | None = None
    batch_size_hessian: int | None = None
    inner_iterations: int | None = None
    update_each: int = 10
    stepsize: float = 1e-2
    use_gamma: bool = True

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
                key=init_params.state.key,
                iter_num=init_params.state.iter_num,
                stepsize=init_params.state.stepsize,
                s_history=init_params.state.s_history,
                y_history=init_params.state.y_history,
                rho_history=init_params.state.rho_history,
                param_history=init_params.state.param_history,
                gamma=init_params.state.gamma,
                last=init_params.state.last,
                param_last=init_params.state.last_param,
            )
            init_params = init_params.params
            dtype = tree_single_dtype(init_params)
        else:
            dtype = tree_single_dtype(init_params)
            state_kwargs = dict(
                key=key,
                iter_num=jnp.asarray(0),
                stepsize=jnp.asarray(self.stepsize, dtype=dtype),
                s_history=init_history(init_params, self.history_size),
                y_history=init_history(init_params, self.history_size),
                rho_history=jnp.zeros(self.history_size, dtype=dtype),
                param_history=init_history(init_params, self.update_each),
                gamma=jnp.asarray(1.0, dtype=dtype),
            )
        (value, aux), grad = self._value_and_grad_with_aux(
            init_params, self.dataset, *args, **kwargs
        )
        return SlbfgsState(
            value=value,
            grad=grad,
            aux=aux,
            error=jnp.asarray(jnp.inf),
            **state_kwargs,
        )

    # def update(
    #     self, params: chex.ArrayTree, state: SlbfgsState, key, *args, **kwargs
    # ) -> jaxopt_base.OptStep:
    #     assert self.inner_iterations is not None
    #     if isinstance(params, jaxopt_base.OptStep):
    #         params = params.params
    #     key = state.key
    #     last = state.last
    #     last_param = state.last_param
    #     s_history = state.s_history
    #     y_history = state.y_history
    #     rho_history = state.rho_history
    #     param_history = state.param_history
    #     inner_params = tree_map(lambda p: p[last_param], param_history)
    #     new_params_sum = tree_map(zeros_like, params)
    #     gamma = state.gamma
    #     loss, grad_f = state.value, state.grad

    #     for k in range(self.inner_iterations):
    #         key, sample_key = random.split(key)
    #         batch = self.draw_batch(sample_key, self.dataset, self.batch_size_gradient)
    #         _, grad_f_x = self._value_and_grad_fun(inner_params, batch, *args, **kwargs)
    #         _, grad_f_p = self._value_and_grad_fun(params, batch, *args, **kwargs)

    #         vrg = tree_add(tree_sub(grad_f_x, grad_f_p), grad_f)

    #         if self.use_gamma:
    #             gamma = compute_gamma(s_history, y_history, last)

    #         d = inv_hessian_product(vrg, s_history, y_history, rho_history, gamma, last)

    #         inner_params = tree_add_scalar_mul(inner_params, -self.stepsize, d)
    #         new_params_sum = tree_add(new_params_sum, inner_params)
    #         last_param = (last_param + 1) % self.update_each
    #         param_history = update_history(param_history, inner_params, last_param)
    #         iters = state.iter_num * self.inner_iterations + k
    #         update = iters > 0 and iters % self.update_each == 0
    #         if update:
    #             old_params = tree_map(lambda p: p[last], state.s_history)
    #             last = (last + 1) % self.history_size
    #             # new_params = tree_reduce(partial(jnp.sum, axis=0), param_history)
    #             # new_params = tree_scalar_mul(1 / self.update_each, new_params)
    #             new_params = param_mean(param_history)
    #             key, sample_key = random.split(state.key)
    #             sr = tree_sub(new_params, old_params)

    #             batch_hessian = self.draw_batch(
    #                 sample_key, self.dataset, self.batch_size_hessian
    #             )

    #             yr = hvp(
    #                 self._value_and_grad_fun,
    #                 [new_params],
    #                 [sr],
    #                 batch_hessian,
    #                 *args,
    #                 has_aux=False,
    #                 value_and_grad=True,
    #                 **kwargs,
    #             )
    #             vdot_sy = tree_vdot(sr, yr)
    #             rho = jnp.where(vdot_sy == 0, 0, 1.0 / vdot_sy)
    #             s_history = update_history(s_history, sr, last)
    #             y_history = update_history(y_history, yr, last)
    #             rho_history = update_history(rho_history, rho, last)

    #     params = tree_scalar_mul(1 / self.inner_iterations, new_params_sum)
    #     (new_loss, aux), grad_f = self._value_and_grad_with_aux(
    #         params, self.dataset, *args, **kwargs
    #     )
    #     error = jnp.abs(new_loss - loss)
    #     state = SlbfgsState(
    #         key=key,
    #         value=new_loss,
    #         grad=grad_f,
    #         error=error,
    #         iter_num=state.iter_num + 1,
    #         stepsize=state.stepsize,
    #         s_history=s_history,
    #         y_history=y_history,
    #         rho_history=rho_history,
    #         param_history=param_history,
    #         gamma=gamma,
    #         aux=aux,
    #         last=last,
    #         last_param=last_param,
    #     )
    #     opt_step = jaxopt_base.OptStep(params=params, state=state)
    #     return opt_step

    def update(
        self, params: chex.ArrayTree, state: SlbfgsState, key, *args, **kwargs
    ) -> jaxopt_base.OptStep:
        if isinstance(params, jaxopt_base.OptStep):
            params = params.params

        iter_state = dict(
            iter_inner=0,
            key=state.key,
            last=state.last,
            last_param=state.last_param,
            s_history=state.s_history,
            y_history=state.y_history,
            rho_history=state.rho_history,
            param_history=state.param_history,
            inner_params=params,
            new_params_sum=tree_map(zeros_like, params),
            grad_f=state.grad,
            gamma=state.gamma,
        )

        def while_body(iter_state):
            iter_inner = iter_state["iter_inner"]
            key = iter_state["key"]
            inner_params = iter_state["inner_params"]
            s_history = iter_state["s_history"]
            y_history = iter_state["y_history"]
            rho_history = iter_state["rho_history"]
            last = iter_state["last"]
            last_param = iter_state["last_param"]
            param_history = iter_state["param_history"]
            new_params_sum = iter_state["new_params_sum"]
            grad_f = iter_state["grad_f"]
            gamma = iter_state["gamma"]

            key, sample_key = random.split(key)
            batch = self.draw_batch(sample_key, self.dataset, self.batch_size_gradient)
            _, grad_f_x = self._value_and_grad_fun(inner_params, batch, *args, **kwargs)
            _, grad_f_p = self._value_and_grad_fun(params, batch, *args, **kwargs)

            vrg = tree_add(tree_sub(grad_f_x, grad_f_p), grad_f)

            if self.use_gamma:
                gamma = compute_gamma(s_history, y_history, last)

            d = inv_hessian_product(vrg, s_history, y_history, rho_history, gamma, last)

            inner_params = tree_add_scalar_mul(inner_params, -self.stepsize, d)
            new_params_sum = tree_add(new_params_sum, inner_params)
            last_param = (last_param + 1) % self.update_each
            param_history = update_history(param_history, inner_params, last_param)
            iters = state.iter_num * self.inner_iterations + iter_inner
            update = (iters > 0) & (iters % self.update_each == 0)

            def update_fun():
                old_params = tree_map(lambda p: p[last], state.s_history)
                new_params = param_mean(param_history)
                new_key, sample_key = random.split(key)
                sr = tree_sub(new_params, old_params)

                batch_hessian = self.draw_batch(
                    sample_key, self.dataset, self.batch_size_hessian
                )

                yr = hvp(
                    self._value_and_grad_fun,
                    [new_params],
                    [sr],
                    batch_hessian,
                    *args,
                    has_aux=False,
                    value_and_grad=True,
                    **kwargs,
                )
                vdot_sy = tree_vdot(sr, yr)
                rho = jnp.where(vdot_sy == 0, 0, 1.0 / vdot_sy)

                return dict(
                    iter_inner=iter_inner + 1,
                    key=new_key,
                    last=(last + 1) % self.history_size,
                    last_param=last_param,
                    s_history=update_history(s_history, sr, last),
                    y_history=update_history(y_history, yr, last),
                    rho_history=update_history(rho_history, rho, last),
                    param_history=param_history,
                    inner_params=inner_params,
                    new_params_sum=new_params_sum,
                    grad_f=state.grad,
                    gamma=gamma,
                )

            def no_update_fun():
                return dict(
                    iter_inner=iter_inner + 1,
                    key=key,
                    last=last,
                    last_param=last_param,
                    s_history=s_history,
                    y_history=y_history,
                    rho_history=rho_history,
                    param_history=param_history,
                    inner_params=inner_params,
                    new_params_sum=new_params_sum,
                    grad_f=state.grad,
                    gamma=gamma,
                )

            return lax.cond(update, update_fun, no_update_fun)

        unroll, jit = self._get_loop_options()
        iter_state = loop.while_loop(
            lambda _: True,
            while_body,
            iter_state,
            self.inner_iterations,
            unroll=unroll,
            jit=jit,
        )
    
        params = tree_scalar_mul(
            1 / self.inner_iterations, iter_state["new_params_sum"]
        )
        (new_loss, aux), grad_f = self._value_and_grad_with_aux(
            params, self.dataset, *args, **kwargs
        )
        error = jnp.abs(new_loss - state.value)
        state = SlbfgsState(
            key=iter_state["key"],
            value=new_loss,
            grad=grad_f,
            error=error,
            iter_num=state.iter_num + 1,
            stepsize=state.stepsize,
            s_history=iter_state["s_history"],
            y_history=iter_state["y_history"],
            rho_history=iter_state["rho_history"],
            param_history=iter_state["param_history"],
            gamma=iter_state["gamma"],
            aux=aux,
            last=iter_state["last"],
            last_param=iter_state["last_param"],
        )
        opt_step = jaxopt_base.OptStep(params=params, state=state)
        return opt_step

    def draw_batch(self, key, dataset, batch_size):
        n = dataset_size(dataset)
        idx = random.choice(key, n, (batch_size,), replace=False)

        return tree_map(lambda d: d[idx], dataset)

    def optimality_fun(self, params, *args, **kwargs):
        return self._value_and_grad_fun(params, *args, **kwargs)[1]

    def _value_and_grad_fun(self, params, *args, **kwargs):
        if isinstance(params, jaxopt_base.OptStep):
            params = params.params
        (value, aux), grad = self._value_and_grad_with_aux(params, *args, **kwargs)
        return value, grad

    def __post_init__(self):
        _, _, self._value_and_grad_with_aux = _make_funs_with_aux(
            fun=self.fun, value_and_grad=self.value_and_grad, has_aux=self.has_aux
        )

        self.dataset_size = dataset_size(self.dataset)
        if self.batch_size_gradient is None:
            self.batch_size_gradient = int(sqrt(self.dataset_size))

        if self.batch_size_hessian is None:
            self.batch_size_hessian = self.batch_size_gradient * self.update_each
        self.batch_size_hessian = min(self.batch_size_hessian, self.dataset_size)

        if self.inner_iterations is None:
            self.inner_iterations = int(self.dataset_size / self.batch_size_gradient)


def param_mean(param_history: chex.ArrayTree) -> chex.ArrayTree:
    n = tree_leaves(tree_map(lambda p: p.shape[0], param_history))
    assert all(_n == n[0] for _n in n)
    history_size = n[0]
    new_params = tree_map(partial(jnp.sum, axis=0), param_history)
    return tree_scalar_mul(1 / history_size, new_params)


def _make_funs_with_aux(fun: Callable, value_and_grad: bool, has_aux: bool):
    if value_and_grad:
        # Case when `fun` is a user-provided `value_and_grad`.

        if has_aux:
            fun_ = lambda *a, **kw: fun(*a, **kw)[0]
            value_and_grad_fun = fun
        else:
            fun_ = lambda *a, **kw: (fun(*a, **kw)[0], None)

            def value_and_grad_fun(*a, **kw):
                v, g = fun(*a, **kw)
                return (v, None), g

    else:
        # Case when `fun` is just a scalar-valued function.
        if has_aux:
            fun_ = fun
        else:
            fun_ = lambda p, *a, **kw: (fun(p, *a, **kw), None)

        value_and_grad_fun = jax.value_and_grad(fun_, has_aux=True)

    def grad_fun(*a, **kw):
        (v, a), g = value_and_grad_fun(*a, **kw)
        return g, a

    return fun_, grad_fun, value_and_grad_fun


# def hvp(f, primals, tangents):
#     return jvp(grad(f), primals, tangents)[1]


def hvp(f, primals, tangents, *args, value_and_grad=False, has_aux=False, **kwargs):
    def grad_f(p):
        if value_and_grad:
            _, _grad_f = f(p, *args, **kwargs)
        else:
            _, _grad_f = jax.value_and_grad(f, has_aux=has_aux)(p, *args, **kwargs)
        return _grad_f

    return jvp(grad_f, primals, tangents)[1]


# todo: two loop recursion with pytrees
# todo: make memories
# todo: RAR
# todo: momentum for x
# todo: other variance reduction strategies
# todo: add tol and break


def inv_hessian_product_leaf(
    v: Array,
    s_history: Array,
    y_history: Array,
    rho_history: Array,
    gamma: Array | float = 1.0,
    start: int = 0,
):
    history_size = len(s_history)

    indices = (start + jnp.arange(history_size)) % history_size

    def body_right(r, i):
        alpha = rho_history[i] * jnp.vdot(s_history[i], r)
        r = r - alpha * y_history[i]
        return r, alpha

    r, alpha = jax.lax.scan(body_right, v, indices, reverse=True)

    r = r * gamma

    def body_left(r, args):
        i, alpha = args
        beta = rho_history[i] * jnp.vdot(y_history[i], r)
        r = r + s_history[i] * (alpha - beta)
        return r, beta

    r, beta = jax.lax.scan(body_left, r, (indices, alpha))

    return r


def inv_hessian_product(
    pytree: chex.ArrayTree,
    s_history: chex.ArrayTree,
    y_history: chex.ArrayTree,
    rho_history: Array,
    gamma: Array | float = 1.0,
    start: int = 0,
):
    """Product between an approximate Hessian inverse and a pytree.

    Histories are pytrees of the same structure as `pytree`.
    Leaves are arrays of shape `(history_size, ...)`, where
    `...` means the same shape as `pytree`'s leaves.

    The notation follows the reference below.

    Args:
      pytree: pytree to multiply with.
      s_history: pytree with the same structure as `pytree`.
        Leaves contain parameter residuals, i.e., `s[k] = x[k+1] - x[k]`.
      y_history: pytree with the same structure as `pytree`.
        Leaves contain gradient residuals, i.e., `y[k] = g[k+1] - g[k]`.
      rho_history: array containing `rho[k] = 1. / vdot(s[k], y[k])`.
      gamma: scalar to use for the initial inverse Hessian approximation,
        i.e., `gamma * I`.
      start: starting index in the circular buffer.

    Reference:
      Jorge Nocedal and Stephen Wright.
      Numerical Optimization, second edition.
      Algorithm 7.4 (page 178).
    """
    fun = partial(
        inv_hessian_product_leaf, rho_history=rho_history, gamma=gamma, start=start
    )
    return tree_map(fun, pytree, s_history, y_history)


def compute_gamma(
    s_history: chex.ArrayTree, y_history: chex.ArrayTree, last: int
) -> Array:
    # Let gamma = vdot(y_history[last], s_history[last]) / sqnorm(y_history[last]).
    # The initial inverse Hessian approximation can be set to gamma * I.
    # See Numerical Optimization, second edition, equation (7.20).
    # Note that unlike BFGS, the initialization can change on every iteration.

    fun1 = lambda s_history, y_history: tree_vdot(y_history[last], s_history[last])
    num = tree_sum(tree_map(fun1, s_history, y_history))
    fun2 = lambda y_history: tree_vdot(y_history[last], y_history[last])
    denom = tree_sum(tree_map(fun2, y_history))
    gamma = jnp.where(denom > 0, num / denom, 1.0)
    assert isinstance(gamma, Array)
    assert gamma.shape == ()
    return gamma


def init_history(pytree, history_size):
    fun = lambda leaf: jnp.zeros((history_size,) + leaf.shape, dtype=leaf.dtype)
    return tree_map(fun, pytree)


def update_history(history_pytree, new_pytree, last):
    fun = lambda history_array, new_value: history_array.at[last].set(new_value)
    return tree_map(fun, history_pytree, new_pytree)


def tree_single_dtype(tree):
    """The dtype for all values in e tree."""
    dtypes = set(p.dtype for p in tree_leaves(tree) if isinstance(p, Array))
    if not dtypes:
        return None
    if len(dtypes) == 1:
        return dtypes.pop()
    raise ValueError("Found more than one dtype in the tree.")
