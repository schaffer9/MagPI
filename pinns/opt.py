from .krr import krr
from .elm import elm

from .prelude import *


Array = ndarray
P = T.ParamSpec('P')
Loss = T.Callable[..., T.Union[Array, tuple[Array, Any]]]
PyTree = Any

#@partial(jit, static_argnames="loss")
def run_batch(
    state: TrainState, 
    loss: Loss, 
    batch: Sequence[Array], 
    **kwargs: Any
) -> tuple[Array, TrainState, PyTree]:
    """Evaluates the loss function for the batch and applies the gradient to
    the ``state``.

    Parameters
    ----------
    state : TrainState
    loss : Loss
        If the loss returns a tuple, the fist value must be the loss and the second 
        value can be a ``PyTree``. This is useful to evaluate some metrics along
        with the loss.
    batch : Sequence[Array]

    Returns
    -------
    tuple[Array, TrainState, Any]
    """
    def _loss(*args: P.args, **kwargs: P.kwargs) -> tuple[Array, Any]:
        out = loss(*args, **kwargs)
        if isinstance(out, tuple):
            return out
        else:
            return out, ()

    (l, aux), grads = jax.value_and_grad(_loss, has_aux=True)(
        state.params, *batch, **kwargs
    )
    return l, state.apply_gradients(grads=grads), aux


#@partial(jit, static_argnames=("loss"))
def run_epoch(
    state: TrainState, 
    loss: Loss, 
    data: Sequence[Sequence[Array]], 
    **loss_kwargs: Any
) -> tuple[TrainState, PyTree]:
    """Runs the optimizer for each batch in the epoch data.

    Parameters
    ----------
    state : TrainState
    loss : Loss
        If the loss returns a tuple, the fist value must be the loss and the second 
        value can be a ``PyTree``. This is useful to evaluate some metrics along
        with the loss.
    data : Sequence[Sequence[Array]]

    Returns
    -------
    tuple[TrainState, PyTree]
    """
    batches = len(data[0])
    def init_metric(m):
        arr = zeros((batches,))
        return arr.at[0].set(m)

    _, state, aux_init = run_batch(state, loss, [d[0] for d in data], **loss_kwargs)
    aux_init = tree_map(init_metric, aux_init)
    def body(batch, state):
        state, aux = state
        batch_data = [d[batch] for d in data]
        _, state, _aux = run_batch(state, loss, batch_data, **loss_kwargs)
        aux = tree_map(lambda m, v: m.at[batch].set(v), aux, _aux)
        return state, aux
    state, aux = lax.fori_loop(1, batches, body, (state, aux_init))
    return state, aux


def make_batches(rng, x, y, batch_size):
    assert len(x) == len(y)
    batches = len(x) // batch_size
    perms = jax.random.permutation(rng, len(x))
    perms = perms[:batches * batch_size]
    perms = perms.reshape((batches, batch_size))
    return x[perms], y[perms]


def train_nn(loss, state, batch_fn, key, epochs=200):
    def init_metric(m):
        arr = zeros((epochs,))
        return arr.at[0].set(mean(m))

    key, subkey = random.split(key)
    state, aux_init = run_epoch(state, loss, batch_fn(subkey))
    aux_init = tree_map(init_metric, aux_init)

    def body(epoch, loop_state):
        state, hist, key = loop_state
        key, subkey = random.split(key)
        batches = batch_fn(subkey)
        state, _aux = run_epoch(state, loss, batches)
        hist = tree_map(lambda m, v: m.at[epoch].set(mean(v)), hist, _aux)
        return state, hist, key


    state, hist, _ = lax.fori_loop(1, epochs, body, (state, aux_init, key))
    return state, hist


def train_nn_ls(state, x, y, key, batch_size=32, epochs=200):
    _loss = vmap(lambda p, x, y: (state.apply_fn(p, x) - y) ** 2, (None, 0, 0))

    def loss(p, x, y):
        l = _loss(p, x, y)
        return l, l

    batch_fn = lambda k: make_batches(k, x, y, batch_size)
    return train_nn(loss, state, batch_fn, key, epochs)