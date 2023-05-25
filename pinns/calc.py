from inspect import signature

from .prelude import *

import chex

# TODO: typing and docs

__all__ = (
    "derivative",
    "value_and_derivative",
    "hvp",
    "curl",
    "cross",
    "divergence",
    "laplace",
)


Scalar = Array


def apply_to_module(operator):
    @wraps(operator)
    def op(f, *args, **kwargs):
        if isinstance(f, nn.Module):
            apply = lambda x, p, *args, **kwargs: f.apply(p, x, *args, **kwargs)
            return operator(apply, *args, **kwargs)
        else:
            return operator(f, *args, **kwargs)

    return op


Partials = int | str


@apply_to_module
def value_and_derivative(
    f: Callable[..., Array], wrt: Partials, argnum: Optional[int] = None
) -> Callable[..., tuple[Array, Array]]:
    sig = signature(f)

    if argnum is not None:
        assert isinstance(argnum, int), "`argnum` must be a integer."
        assert isinstance(wrt, int), "`wrt` must be a integer if argnum is given."

    if isinstance(wrt, str):
        arg_names = list(sig.parameters.keys())
        assert (
            wrt in arg_names
        ), f"Argument `{wrt}` is not a valid attribute name for the function."
        wrt = arg_names.index(wrt)

    def df(*args, **kwargs):
        if isinstance(wrt, int) and argnum is None:
            ei = [
                ones_like(x) if i == wrt else zeros_like(x) for i, x in enumerate(args)
            ]
        elif isinstance(wrt, int) and argnum is not None:
            ei = [zeros_like(x) for x in args]
            ei[argnum] = ei[argnum].at[wrt].set(1.0)
        else:
            ei = wrt
        _f = lambda *args: f(*args, **kwargs)
        primals, tangents = jax.jvp(_f, list(args), ei)
        return primals, tangents

    return df


@apply_to_module
def derivative(
    wrt: Callable[..., Array] | Partials | Sequence[Partials],
    argnums: T.Optional[int | Sequence[int | None]] = None,
) -> Callable[..., Callable | Array]:
    """This derivative operator computes the respective derivatives
    using forward mode differentiation.

    >>> f = lambda x, y: x ** 2 + y
    >>> df = derivative(f, 0)(3., 1.)
    >>> bool(jnp.isclose(df, 6.))
    True
    >>> f = lambda x, y: x ** 2 * y + x * y ** 2
    >>> df = derivative(['x', 'y'])(f)(3., 5.)
    >>> bool(jnp.isclose(df, 16.))
    True
    >>> g = lambda x, y: sin(x * y**2)
    >>> dg = jit(derivative(g, 1))(1., 0.5)
    >>> bool(jnp.isclose(dg, cos(0.5 ** 2)))
    True
    >>> g = lambda x, y: sin(y[0]) * cos(y[1]) * x
    >>> dg = derivative([1], argnums=[1])(g)(1., array([0.5, 1.0]))
    >>> bool(jnp.isclose(dg, -sin(0.5) * sin(1.)))
    True
    >>> g = lambda x, y: sin(x) @ cos(y)
    >>> dg = derivative([0, 0], argnums=[0, 1])(g)(array([0.5, 1.0]), array([0.6, 1.0]))
    >>> bool(jnp.isclose(dg, -cos(0.5) * sin(0.6)))
    True

    Parameters
    ----------
    wrt : Union[Callable[..., Array], int, str, Sequence[Union[int, str]]]
        function, argnums, string or list of partial derivatives to compute
    argnums : Optional[int], optional
        if provided, the respective argument is considered a vector `wrt` should indicate the index
        of the partial derivative to compute

    Returns
    -------
    Callable[..., Array]
    """
    if callable(wrt):
        msg = "If a function is passed directly, `argnum` must be None or an integer"
        assert not isinstance(argnums, Sequence), msg
        if argnums is None:
            argnum = 0
        elif isinstance(argnums, Sequence):
            assert len(argnums) == 1
            argnum = argnums[0]
        else:
            argnum = argnums
        f = wrt
        sig = signature(f)
        df = value_and_derivative(f, argnum)
        _df = lambda *args, **kwargs: df(*args, **kwargs)[1]
        setattr(_df, "__signature__", sig)
        return _df

    if isinstance(wrt, int):
        wrt = [wrt]

    if not isinstance(argnums, Sequence):
        argnums = [argnums for _ in wrt]

    assert len(argnums) == len(wrt)

    def inner(f) -> Callable[..., Array]:
        def _df(f, ei, argnum):
            sig = signature(f)
            df = value_and_derivative(f, ei, argnum=argnum)
            derivative_f = lambda *args, **kwargs: df(*args, **kwargs)[1]
            setattr(derivative_f, "__signature__", sig)
            return derivative_f

        df = f
        assert isinstance(wrt, Sequence)
        assert isinstance(argnums, Sequence)
        for ei, argnum in zip(wrt, argnums):
            df = _df(df, ei, argnum)
        return df

    return inner


def hvp(
    f: Callable[..., Array], primals: Sequence[Array], tangents: Sequence[Array]
) -> Array:
    return jvp(jacfwd(f), primals, tangents)[1]


def hvp_forward_over_reverse(
    f: Callable,
    primals: Sequence[chex.ArrayTree],
    tangents: Sequence[chex.ArrayTree],
    *args: Any,
    value_and_grad: bool = False,
    has_aux: bool = False,
    **kwargs: Any,
):
    """Computes the hessian vector product of a Scalar valued
    function in forward-over-reverse mode.

    Args:
        f (Callable):
        primals Sequence[chex.ArrayTree]:
        tangents Sequence[chex.ArrayTree]:
        value_and_grad (bool, optional): Defaults to False.
        has_aux (bool, optional): Defaults to False.
    """

    def grad_f(p):
        if value_and_grad:
            _, _grad_f = f(p, *args, **kwargs)
        else:
            _, _grad_f = jax.value_and_grad(f, has_aux=has_aux)(p, *args, **kwargs)
        return _grad_f

    return jvp(grad_f, primals, tangents)[1]


def hessian_diag(f: Callable[..., Array], primals: Array) -> Array:
    primals = asarray(primals)
    primals = primals
    is_1d = primals.shape == ()
    primals = primals.ravel()

    vs = jnp.eye(primals.shape[0])
    comp = lambda v: tree_map(lambda a: a @ v, hvp(f, [primals], [v]))
    diag_entries = jax.vmap(comp)(vs)
    if is_1d:
        return diag_entries[0]
    else:
        return diag_entries


@apply_to_module
def laplace(f: Callable[..., Scalar]) -> Callable[..., Scalar]:
    """Computes the laplacian :math:`\\Delta f` wrt. the first argument.
    If `f` is a vector valued function, the laplacian of each output is
    computed.

    Parameters
    ----------
    f : Callable[..., Scalar]
    """

    def lap(x, *args, **kwargs):
        H_diag = hessian_diag(lambda x: f(x, *args, **kwargs), x)
        return tree_map(lambda d: jnp.sum(d, axis=0), H_diag)

    return lap


@apply_to_module
def divergence(f: Callable[..., Array]) -> Callable[..., Array]:
    """Computes the divergence :math:`\\nabla \\cdot f` wrt. the first argument.

    Parameters
    ----------
    f : Callable[..., Array]
    """

    def div_f(x, *args, **kwargs):
        Jf = jacfwd(f, 0)(x, *args, **kwargs)
        return jnp.sum(diag(Jf))

    return div_f


@apply_to_module
def curl(f: Callable[..., Array]) -> Callable[..., Array]:
    """Computes the curl :math:`\\nabla \\times f` wrt. the first argument.

    Parameters
    ----------
    f : Callable[..., Array]
    """

    def _curl(x, *args):
        if x.shape[0] == 2:
            return curl2d(f)(x, *args)
        elif x.shape[0] == 3:
            return curl3d(f)(x, *args)
        else:
            raise ValueError("Dimension must be 2 or 3.")

    return _curl


def curl3d(f):
    def _f(
        x,
        *args,
    ):
        J = jacfwd(f, 0)(x, *args)
        x = J[2, 1] - J[1, 2]
        y = J[0, 2] - J[2, 0]
        z = J[1, 0] - J[0, 1]
        c = jnp.stack((x, y, z))
        return c

    return _f


def curl2d(f):
    def _f(x, *args):
        J = jacfwd(f, 0)(x, *args)
        c = J[1, 0] - J[0, 1]
        return c

    return jit(_f)


def is_2d(*args) -> bool:
    return all(map(lambda x: x.shape[-1] == 2, args))


def is_3d(*args) -> bool:
    return all(map(lambda x: x.shape[-1] == 3, args))
