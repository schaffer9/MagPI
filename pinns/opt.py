
from .prelude import *

Array = ndarray
P = T.ParamSpec('P')
Loss = T.Callable[P, T.Union[Array, tuple[Array, Any]]]


@partial(jit, static_argnames="loss")
def run_batch(
    state: TrainState, 
    loss: Loss, 
    batch: Sequence[Array], 
    **kwargs: Any
) -> tuple[Array, TrainState, Any]:
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
        # if not isinstance(out, ndarray):
        #     return out
        # else:
        #     return out, ()
    (l, aux), grads = jax.value_and_grad(_loss, has_aux=True)(
        {'params': state.params}, *batch, **kwargs
    )
    return l, state.apply_gradients(grads=grads['params']), aux


@partial(jit, static_argnames=("loss"))
def run_epoch(
    state: TrainState, 
    loss: Loss, 
    data: Sequence[Sequence[Array]], 
    **loss_kwargs: Any
) -> Union[TrainState, tuple[TrainState, Any]]:
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
    Union[TrainState, tuple[TrainState, Any]]
        A tuple is only returned if the loss returns some additional metrics.
        The metric is returned for each batch.
    """
    batches = len(data[0])
    def init_metric(m):
        arr = zeros((batches,))
        return arr.at[0].set(m)

    _, state, aux_init = run_batch(state, loss, [d[0] for d in data], **loss_kwargs)
    aux_init = tree_map(init_metric, aux_init)
    has_aux = not (isinstance(aux_init, tuple) and len(aux_init) == 0)
    def body(batch, state):
        state, aux = state
        batch_data = [d[batch] for d in data]
        _, state, _aux = run_batch(state, loss, batch_data, **loss_kwargs)
        aux = tree_map(lambda m, v: m.at[batch].set(v), aux, _aux)
        return state, aux
    state, aux = lax.fori_loop(1, batches, body, (state, aux_init))
    if has_aux:
        return state, aux
    else:
        return state
