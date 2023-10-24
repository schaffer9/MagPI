from .prelude import *
from .r_fun import ADF
from .calc import value_and_jacfwd


Collection = T.Mapping[str, Array]
Params = T.Mapping[str, Collection]
Activation = Callable[[Array], Array]
Model = Callable[..., Array]


class MLP(nn.Module):
    layers: Sequence[int]
    activation: Optional[Activation] = None

    @nn.compact
    def __call__(self, x):
        if self.activation is None:
            activation = tanh
        else:
            activation = self.activation

        for i, layer in enumerate(self.layers[:-1]):
            x = activation(nn.Dense(layer, name=f"layers_{i}")(x))

        output_neurons = self.layers[-1]
        x = nn.Dense(output_neurons, name="output_layer")(x)
        if output_neurons == 1:
            return x[..., 0]
        else:
            return x


def mlp(
    key: Array, layers: Sequence[int], activation: Optional[Activation] = None
) -> tuple[MLP, Params]:
    """Creates a Multi Layer Perceptron with the given layers and activation function.

    Examples
    --------
    >>> key1, key2 = random.split(random.PRNGKey(42), 2)

    This creates a MLP with two input features, two hidden layers and a scalar output.

    >>> model, params = mlp(key1, [2, 10, 10, 1], swish)
    >>> x = random.uniform(key2, (10, 2))
    >>> y = model.apply(params, x)

    Parameters
    ----------
    key : Array
        Random key for initialization
    layers : Sequence[int]
        Number of neurons in each layer. This must include input and output layer.
    activation : Optional[Activation], defaults to ``tanh``

    Returns
    -------
    tuple[MLP, flax.core.FrozenDict]
    """
    features = layers[0]
    key, init_key = random.split(key, 2)
    model = MLP(layers[1:], activation)
    x_init = random.uniform(key, (features,))
    params = model.init(init_key, x_init)
    return (model, params)


def impose_ic(
    m0: Model,
    model: Model | None = None,
    t0=0.0,
    decay_rate=10.0,
    argnums_m0: int | Sequence[int] = 0,
    argnum_t: int = 1,
) -> Callable:
    """Imposes the prescribed initial conditions given by m0.
    This follows an exponential decay `exp(-decay_rate*(t-t0)) * m0 + (1-exp(-decay_rate*(t-t0))) * model`

    Args:
        m0 (Model): initial condition model
        model (Model | None, optional): free model. Defaults to None.
        t0 (_type_, optional): initial time. Defaults to 0..
        decay_rate (_type_, optional): exponential decay rate. Defaults to 10..
        argnums_m0 (int | Sequence[int], optional):
            specifies which positional arguments should be taken into account for m0.
            Defaults to 0.
        argnum_t (int, optional):
            specifies which positional arguments represents time `t`.
            Defaults to 1.

    Returns:
        Callable: Model or decorator
    """
    if isinstance(argnums_m0, int):
        argnums_m0 = [argnums_m0]

    def impose(model):
        def inner(*args, **kwargs):
            assert not isinstance(argnums_m0, int)
            assert model is not None
            args_m0 = (args[i] for i in argnums_m0)
            t = args[argnum_t]
            decay = jnp.exp(-decay_rate * (t - t0))
            value = model(*args, **kwargs)
            return m0(*args_m0) * decay + (1 - decay) * value

        return inner

    if model is None:
        return impose
    else:
        return impose(model)


def impose_dirichlet_bc(
    adf: ADF,
    model: Model | None = None,
    g: Callable[..., Array] | None = None,
    argnums_adf: int | Sequence[int] = 0,
    argnums_g: int | Sequence[int] = 0,
) -> Callable:
    """Imposes Dirichlet boundary conditions `g` onto the model.
    If `g` is not given, homogenious zero boundary conditions are assumed.

    Args:
        adf (ADF): 1st order normalized approximate distance function.
        g (Callable[..., Array] | None, optional): prescribed boundary conditions. Defaults to None.
        model (Model | None, optional):
            Unconstrained model. If not provided, the function acts as a decorator.
            Defaults to None.
        argnums_adf (int | Sequence[int], optional):
            specifies which positional arguments are passed to `adf`. Defaults to 0.
        argnums_g (int | Sequence[int], optional):
            specifies which positional arguments are passed to `g`.
            Defaults to 0.

    Returns:
        Callable: a new model with the exact imposition of the prescribed
        boundary conditions or a decorator for such a model.
    """
    if g is None:
        _h = lambda *a, **k: asarray(0.0)
    else:
        _h = g

    if isinstance(argnums_adf, int):
        argnums_adf = [argnums_adf]

    if isinstance(argnums_g, int):
        argnums_g = [argnums_g]

    def impose(model):
        def _model(*args, **kwargs):
            assert not isinstance(argnums_adf, int)
            assert not isinstance(argnums_g, int)
            x_adf = (args[i] for i in argnums_adf)
            l = adf(*x_adf)
            value = model(*args, **kwargs)
            x_g = (args[i] for i in argnums_g)
            value_g = _h(*x_g)
            return l * value + value_g

        return _model

    if model is None:
        return impose
    else:
        return impose(model)


def impose_neumann_bc(
    adf: ADF,
    model: Model | None = None,
    h: Callable[..., Array] | None = None,
    argnums_adf: int | Sequence[int] = 0,
    argnum_model: int = 0,
    argnums_h: int | Sequence[int] = 0,
) -> Callable:
    """Imposes Neumann boundary conditions `h` onto the model.
    If `h` is not given, homogenious zero boundary conditions are assumed.

    Args:
        adf (ADF): 1st order normalized approximate distance function.
        model (Model | None, optional):
            Unconstrained model. If not provided, the function acts as a decorator.
            Defaults to None.
        h (Callable[..., Array] | None, optional): prescribed boundary conditions. Defaults to None.
        argnums_adf (int | Sequence[int], optional):
            specifies which positional arguments are passed to `adf`. Defaults to 0.
        argnum_model (int, optional):
            specifies which positional argument which is used to compute the normal derivative. Defaults to 0.
        argnums_h (int | Sequence[int], optional):
            specifies which positional arguments are passed to `h`.
            Defaults to 0.

    Returns:
        Callable: a new model with the exact imposition of the prescribed
        boundary conditions or a decorator for such a model.
    """
    if h is None:
        _h = lambda *a, **k: asarray(0.0)
    else:
        _h = h

    if isinstance(argnums_adf, int):
        argnums_adf = [argnums_adf]

    if isinstance(argnums_h, int):
        argnums_h = [argnums_h]

    def impose(model):
        def _model(*args, **kwargs):
            assert not isinstance(argnums_adf, int)
            assert not isinstance(argnums_h, int)
            x = args[argnum_model]
            f = lambda x: model(
                *args[:argnum_model], x, *args[argnum_model + 1 :], **kwargs
            )
            x_adf = (args[i] for i in argnums_adf)
            l, n = value_and_jacfwd(adf)(*x_adf)
            value, normal_derivative = jvp(f, [x], [n])
            x_h = (args[i] for i in argnums_h)
            value_h = _h(*x_h)

            return value - l * normal_derivative - l * value_h

        return _model

    if model is None:
        return impose
    else:
        return impose(model)
