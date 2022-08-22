from inspect import signature

from .prelude import *

# TODO: typing and docs

__all__ = (
    'derivative', 'value_and_derivative', 'hvp', 'curl', 'cross', 'divergence', 'laplace'
)


Array = Any
Scalar = Array


def apply_to_module(operator):
    @wraps(operator)
    def op(f, *args, **kwargs):
        if isinstance(f, Module):
            apply = lambda x, p, *args, **kwargs: f.apply(p, x, *args, **kwargs)
            return operator(apply, *args, **kwargs)
        else:
            return operator(f, *args, **kwargs)
    return op


Partials = int | str

@apply_to_module
def value_and_derivative(
    f: Callable[..., Array], 
    wrt: Partials, 
    argnum: Optional[int]=None
) -> Callable[..., tuple[Array, Array]]:
    sig = signature(f)

    if argnum is not None:
        assert isinstance(argnum, int), "`argnum` must be a integer."
        assert isinstance(wrt, int), "`wrt` must be a integer if argnum is given."

    if isinstance(wrt, str):
        arg_names = list(sig.parameters.keys())
        assert wrt in arg_names, f"Argument `{wrt}` is not a valid attribute name for the function."
        wrt = arg_names.index(wrt)

    def df(*args, **kwargs):
        if isinstance(wrt, int) and argnum is None:
            ei = [ones_like(x) if i == wrt else zeros_like(x) for i, x in enumerate(args)]
        elif isinstance(wrt, int) and argnum is not None:
            ei = [zeros_like(x) for x in args]
            ei[argnum] = ei[argnum].at[wrt].set(1.)
        else:
            ei = wrt
        _f = lambda *args: f(*args, **kwargs)
        primals, tangents = jax.jvp(_f, list(args), ei)
        return primals, tangents
    return df


@apply_to_module
def derivative(
    wrt: Callable[..., Array] | Partials | Sequence[Partials], 
    argnums: T.Optional[int | Sequence[int | None]]=None
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


def _save_grad(f):
    def g(*args, **kwargs):
        y = f(*args, **kwargs)
        output_is_scalar = len(y.shape) == 0
        output_is_1d = len(y.shape) == 1 and y.shape[0] == 1
        assert output_is_1d or output_is_scalar, "Output of function must be a scalar"
        if output_is_1d:
            return y[0]
        else:
            return y
    return grad(g)


def hvp(
    f: Callable[..., Array], 
    primals: Sequence[Array], 
    tangents: Sequence[Array]) -> Array:
    return jvp(_save_grad(f), primals, tangents)[1]


def hessian_diag(f: Callable[..., Array], primals: Array) -> Array:
    assert len(primals.shape) == 1
    vs = jnp.eye(primals.shape[0])
    comp = lambda v: vdot(v, hvp(f, [primals], [v]))
    return jax.vmap(comp)(vs)


@apply_to_module
def laplace(f: Callable[..., Scalar]) -> Callable[..., Scalar]:
    """Computes the laplacian :math:`\\Delta f` wrt. the first argument.

    Parameters
    ----------
    f : Callable[..., Scalar]
    """
    def lap(x, *args, **kwargs):
        H_diag = hessian_diag(lambda x: f(x, *args, **kwargs), x)
        return jnp.sum(H_diag)
    
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
    def _f(x, *args, ):
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


def cross(a: Array, b: Array) -> Array:
    """Computes the cross product of two vectors in :math:`\\mathbb{R}^2` or :math:`\\mathbb{R}^3`.

    Parameters
    ----------
    a : Array
    b : Array

    Returns
    -------
    Array
    """
    if is_2d(a, b):
        return cross2d(a, b)
    elif is_3d(a, b):
        return cross3d(a, b)
    else:
        dim_a = a.shape
        dim_b = b.shape
        raise ValueError(f"Cannot build cross product for dim {dim_a} and {dim_b}.")

def cross2d(a: Array, b: Array) -> Array:
    return a[0] * b[1] - a[1] * b[0]


def cross3d(a: Array, b: Array) -> Array:
    x = a[1] * b[2] - b[1] * a[2]
    y = b[0] * a[2] - a[0] * b[2]
    z = a[0] * b[1] - b[0] * a[1]
    return stack((x, y, z))


def is_2d(*args) -> bool:
    return all(map(lambda x: x.shape[-1] == 2, args))


def is_3d(*args) -> bool:
    return all(map(lambda x: x.shape[-1] == 3, args))
