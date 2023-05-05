from .prelude import *

Collection = T.Mapping[str, Array]
Params = T.Mapping[str, Collection]
Activation = Callable[[Array], Array]


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
            x = activation(nn.Dense(layer, name=f'layers_{i}')(x))
        
        output_neurons = self.layers[-1]
        x = nn.Dense(output_neurons, name='output_layer')(x)
        if output_neurons == 1:
            return x[..., 0]
        else:
            return x


def mlp(
    key: random.KeyArray,
    layers: Sequence[int],
    activation: Optional[Activation] = None
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
    key : KeyArray
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
    x_init = random.uniform(key, (features, ))
    params = model.init(init_key, x_init)
    return (model, params)
