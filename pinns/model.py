from .prelude import *

Array = ndarray
Collection = T.Mapping[str, Array]
Params = T.Mapping[str, Collection]


class MLP(Module):
    layers: Sequence[int]
    activation: Callable = tanh

    @compact
    def __call__(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = tanh(Dense(layer, name=f'layers_{i}')(x))
        
        output_neurons = self.layers[-1]
        x = Dense(output_neurons, name='output_layer')(x)
        if output_neurons == 1:
            return x[..., 0]
        else:
            return x

    def apply(self, variables: Params, *args: Array) -> Array:
        output = super().apply(variables, *args)
        if isinstance(output, tuple):
            return output[0]
        else:
            return output


def mlp(
    key: random.KeyArray, 
    layers: Sequence[int], 
    activation: Callable[[Array], Array] = tanh
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
    key : ndarray
        Random key for initialization
    layers : Sequence[int]
        Number of neurons in each layer. This must include input and output layer.
    activation : Callable, optional
        Activation function, by default tanh

    
    Returns
    -------
    tuple[MLP, flax.core.FrozenDict]
    """
    features = layers[0]
    key, init_key = random.split(key, 2)
    model = MLP(layers[1:], activation)
    x_init = random.uniform(key, (features, ))
    params = model.init(init_key, x_init)
    return (model, params)