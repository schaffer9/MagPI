from .prelude import *


# some function taken from jaxopt:

def make_funs_with_aux(fun: Callable, value_and_grad: bool, has_aux: bool):
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


def tree_single_dtype(tree):
    """The dtype for all values in e tree."""
    dtypes = set(p.dtype for p in tree_leaves(tree) if isinstance(p, Array))
    if not dtypes:
        return None
    if len(dtypes) == 1:
        return dtypes.pop()
    raise ValueError("Found more than one dtype in the tree.")
