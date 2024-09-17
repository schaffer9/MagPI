from inspect import signature
from typing import Any, Callable, Sequence, Optional, Literal, overload

from chex import ArrayTree

from .prelude import *

# TODO: typing and docs

__all__ = (
    "derivative",
    "value_and_derivative",
    "hvp",
    "curl",
    "divergence",
    "laplace",
    "value_and_jacfwd",
)


Scalar = Array
PyTree = ArrayTree


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
    argnums: Optional[int | Sequence[int | None]] = None,
) -> Callable[..., Callable | PyTree]:
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


def value_and_jacfwd(
    f: Callable,
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
) -> Callable:
    def _f(*args, **kwargs):
        y = f(*args, **kwargs)
        if has_aux:
            y, aux = y
            return y, (y, aux)
        else:
            return y, y

    def df(*args, **kwargs):
        J, y = jacfwd(_f, argnums, has_aux=True, holomorphic=holomorphic)(
            *args, **kwargs
        )
        return y, J

    return df


def value_and_jacrev(
    f: Callable,
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
) -> Callable:
    def _f(*args, **kwargs):
        y = f(*args, **kwargs)
        if has_aux:
            y, aux = y
            return y, (y, aux)
        else:
            return y, y

    def df(*args, **kwargs):
        J, y = jacrev(
            _f, argnums, has_aux=True, holomorphic=holomorphic, allow_int=allow_int
        )(*args, **kwargs)
        return y, J

    return df


def hvp(
    f: Callable[..., PyTree], primals: Sequence[PyTree], tangents: Sequence[PyTree]
) -> PyTree:
    return jvp(jacfwd(f), primals, tangents)[1]


def hvp_forward_over_reverse(
    f: Callable,
    primals: Sequence[PyTree],
    tangents: Sequence[PyTree],
    *args: Any,
    alpha: Optional[float | Array] = None,
    value_and_grad: bool = False,
    has_aux: bool = False,
    **kwargs: Any,
) -> PyTree:
    """Computes the hessian vector product of a Scalar valued
    function in forward-over-reverse mode.

    Args:
        f (Callable):
        primals Sequence[PyTree]:
        tangents Sequence[PyTree]:
        value_and_grad (bool, optional): Defaults to False.
        has_aux (bool, optional): Defaults to False.
    """

    def grad_f(*p):
        if value_and_grad:
            _, _grad_f = f(*p, *args, **kwargs)
        else:
            _, _grad_f = jax.value_and_grad(f, has_aux=has_aux)(*p, *args, **kwargs)
        return _grad_f

    _hvp = jvp(grad_f, primals, tangents)[1]
    if alpha is None:
        return _hvp
    else:
        return tree_add(_hvp, tree_scalar_mul(alpha, tangents[0]))


HVP = Callable


@overload
def value_grad_hvp(
    f: Callable,
    primals: PyTree,
    *args: Any,
    value_and_grad: bool = False,
    has_aux: Literal[False] = False,
    **kwargs: Any,
) -> tuple[Array, PyTree, HVP]: ...


@overload
def value_grad_hvp(
    f: Callable,
    primals: PyTree,
    *args: Any,
    value_and_grad: bool = False,
    has_aux: Literal[True] = True,
    **kwargs: Any,
) -> tuple[Array, PyTree, HVP, Any]: ...


def value_grad_hvp(
    f: Callable,
    primals: PyTree,
    *args: Any,
    value_and_grad: bool = False,
    has_aux: bool = False,
    **kwargs: Any,
) -> tuple[Array, PyTree, HVP] | tuple[Array, PyTree, HVP, Any]:
    """Computes the value, gradient and hvp of a function.

    Args:
        f (Callable):
        primals PyTree:
        value_and_grad (bool, optional): Defaults to False.
        has_aux (bool, optional): Defaults to False.

    Returns:
        If `has_aux` is `False`, it returns a tuple
        `(f(*primals), grad(f)(*primals), hvp)`. If `has_aux` is
        `True` it returns `(f(*primals), grad(f)(*primals), hvp, aux)`.

    """

    def grad_f(p: PyTree) -> PyTree:
        if value_and_grad:
            value, _grad_f = f(p, *args, **kwargs)
        else:
            value, _grad_f = jax.value_and_grad(f, has_aux=has_aux)(p, *args, **kwargs)
        return _grad_f, value

    _grad_f, hvp, value = jax.linearize(grad_f, primals, has_aux=True)
    if has_aux:
        value, aux = value
        return value, _grad_f, hvp, aux
    else:
        return value, _grad_f, hvp


def hvp_reverse_over_reverse(
    f: Callable,
    primals: Sequence[PyTree],
    tangents: Sequence[PyTree],
    *args: Any,
    value_and_grad: bool = False,
    has_aux: bool = False,
    **kwargs: Any,
) -> PyTree:
    def grad_f(p):
        if value_and_grad:
            _, _grad_f = f(p, *args, **kwargs)
        else:
            _, _grad_f = jax.value_and_grad(f, has_aux=has_aux)(p, *args, **kwargs)
        return _grad_f

    (x,) = primals
    (v,) = tangents
    return grad(lambda x: tree_vdot(grad_f(x), v))(x)


def hessian_diag(f: Callable[..., PyTree], primals: Array) -> PyTree:
    primals = asarray(primals)
    is_1d = primals.shape == ()
    
    if is_1d:
        v = asarray(1.0)
        diag_entrie = hvp(f, [primals], [v])
        return tree_map(lambda t: asarray([t]), diag_entrie)
    else:
        primals = primals.ravel()
        vs = jnp.eye(primals.shape[0])

        def comp(v):
            return tree_map(lambda a: a @ v, hvp(f, [primals], [v]))

        diag_entries = jax.vmap(comp)(vs)
        return diag_entries


@apply_to_module
def laplace(f: Callable[..., PyTree]) -> Callable:
    """Computes the laplacian :math:`\\Delta f` wrt. the first argument.
    If `f` is a vector valued function, the laplacian of each output is
    computed.

    Parameters
    ----------
    f : Callable
    """

    def lap(x, *args, **kwargs):
        H_diag = hessian_diag(lambda x: f(x, *args, **kwargs), x)
        return tree_map(lambda d: jnp.sum(d, axis=0), H_diag)

    return lap


@apply_to_module
def divergence(f: Callable[..., PyTree], argnums=0) -> Callable[..., PyTree]:
    """Computes the divergence :math:`\\nabla \\cdot f` wrt. the first argument.

    Parameters
    ----------
    f : Callable[..., Array]
    """

    def div_f(*args, **kwargs):
        _, d = value_and_divergence(f, argnums)(*args, **kwargs)
        return d

    return div_f


@apply_to_module
def value_and_divergence(
    f: Callable[..., PyTree], argnums=0
) -> Callable[..., tuple[PyTree, PyTree]]:
    """Computes the divergence :math:`\\nabla \\cdot f` wrt. the first argument.

    Parameters
    ----------
    f : Callable[..., Array]
    """

    def div_f(*args, **kwargs):
        y, Jf = value_and_jacfwd(f, argnums)(*args, **kwargs)

        def _compute_div(Jf):
            if len(Jf.shape) > 2:
                return jnp.sum(vmap(diag, 0)(Jf), -1)
            else:
                return jnp.sum(diag(Jf))

        return y, tree_map(_compute_div, Jf)

    return div_f


@apply_to_module
def curl(f: Callable[..., PyTree]) -> Callable[..., PyTree]:
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
    def _f(x, *args):
        def h(x, *args):
            v = f(x, *args)
            return v, v

        J, v = jacfwd(h, 0, has_aux=True)(x, *args)
        assert is_3d(v), "Function must have 3d output"

        def _curl(J):
            x = J[..., 2, 1] - J[..., 1, 2]
            y = J[..., 0, 2] - J[..., 2, 0]
            z = J[..., 1, 0] - J[..., 0, 1]
            c = jnp.stack((x, y, z), axis=-1)
            return c

        return tree_map(_curl, J)

    return _f


def curl2d(f):
    def _f(x, *args):
        def h(x, *args):
            v = f(x, *args)
            return v, v

        J, v = jacfwd(h, 0, has_aux=True)(x, *args)
        assert is_2d(v), "Function must have 2d output"

        def _curl(J):
            c = J[..., 1, 0] - J[..., 0, 1]
            return c

        return tree_map(_curl, J)

    return jit(_f)


def is_2d(*args) -> bool:
    return all(tree_leaves(tree_map(lambda x: x.shape[-1] == 2, args)))


def is_3d(*args) -> bool:
    return all(tree_leaves(tree_map(lambda x: x.shape[-1] == 3, args)))
