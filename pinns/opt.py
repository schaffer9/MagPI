import dataclasses

import chex
from flax.struct import dataclass, field
from flax.training import train_state

from pinns.prelude import *


P = T.ParamSpec("P")
PyTree = chex.ArrayTree
Metric = Array | tuple[Array, ...] | dict[str, Array]
Params = PyTree
Epoch = int
Loss = T.Callable[..., T.Union[Array, tuple[Array, Metric]]]


class State(T.Protocol):
    params: Params
    tx: optax.GradientTransformation
    opt_state: optax.OptState

    def replace(self, /, **changes) -> "State":
        ...


@dataclass
class Batch:
    batch_number: int
    batches: int
    data: PyTree = field(repr=False)

    def __getitem__(self, item):
        assert hasattr(
            self.data, "__getitem__"
        ), "`data` has not impolemented `__getitem__`."
        return self.data.__getitem__(item)


BatchFn = Callable[[random.KeyArray, State], Sequence[Batch]]


class TrainStep(T.Protocol):
    def __call__(
        self, state: State, batch: Batch, **kwargs: Any
    ) -> tuple[State, Array, Metric]:
        ...


def step_fn(loss: Loss):
    def train_step(
        state: train_state.TrainState, batch: Batch
    ) -> tuple[train_state.TrainState, Array, Metric]:
        def _loss(*args: P.args, **kwargs: P.kwargs) -> tuple[Array, Metric]:
            _l = loss(*args, **kwargs)
            if isinstance(_l, tuple):
                return _l
            else:
                return _l, {"loss": _l}

        (l, aux), grads = jax.value_and_grad(_loss, has_aux=True)(
            state.params, *batch.data
        )
        return state.apply_gradients(grads=grads), l, aux

    return train_step


@partial(jit, static_argnames=("train_step"))
def run_epoch(
    state: State,
    train_step: TrainStep,
    data: Sequence[PyTree],
) -> tuple[State, PyTree]:
    """Runs the optimizer for each batch in the epoch data.

    Parameters
    ----------
    state : TrainState
    train_step : TrainStep
    data : Sequence[Sequence[Array]]
    **train_step_kwargs : Any

    Returns
    -------
    tuple[TrainState, PyTree]
    """
    # msg = "All inputs must have the same amount of batches along the first axis."
    # assert check_axis_size(data), msg
    # all_batches = tree_map(lambda p: p.shape[0], data)
    # _batches = tree_leaves(all_batches)
    # msg = "All inputs must have the same amount of batches along the first axis."
    # for b in _batches:
    #     assert b == _batches[0], msg
    # batches = _batches[0]
    batches = axis_size(data)

    def init_metric(m):
        arr = zeros((batches,))
        return arr.at[0].set(m)

    _batch = Batch(0, batches, tree_map(lambda b: b[0], data))
    state, loss, aux_init = train_step(state, _batch)
    aux_init = tree_map(init_metric, (loss, aux_init))

    def body(batch_number, state):
        state, aux = state
        batch_data = tree_map(lambda b: b[batch_number], data)
        _batch = Batch(batch_number, batches, batch_data)
        state, loss, _aux = train_step(state, _batch)
        aux = tree_map(lambda m, v: m.at[batch_number].set(v), aux, (loss, _aux))
        if hasattr(state, "on_batch_end"):
            state = state.on_batch_end(loss, _aux)
        return state, aux

    state, aux = lax.fori_loop(1, batches, body, (state, aux_init))
    return state, aux


def check_axis_size(tree: PyTree, axis: int = 0) -> bool:
    axis_size = tree_map(lambda p: p.shape[axis], tree)
    axis_size = tree_leaves(axis_size)
    s = axis_size[0]
    return all(_s == s for _s in axis_size)


def axis_size(tree: PyTree, axis: int = 0) -> bool:
    msg = f"All inputs must have the same amount of batches along axis {axis}."
    assert check_axis_size(tree, axis), msg
    axis_size = tree_map(lambda p: p.shape[axis], tree)
    axis_size = tree_leaves(axis_size)
    return axis_size[0]


def batches_without_replacement(
    key: random.KeyArray, X: PyTree, batch_size: int
) -> PyTree:
    N = axis_size(X)
    batches = N // batch_size
    perms = jax.random.permutation(key, N)
    perms = perms[: batches * batch_size]
    perms = perms.reshape((batches, batch_size))
    return tree_map(lambda x: x[perms], X)


def batches_with_replacement(
    key: random.KeyArray,
    X: PyTree,
    batches: int,
    batch_size: int,
    replace_within_batch=True,
) -> PyTree:
    N = axis_size(X)

    def choose_batch(key, batch_size):
        return random.choice(key, N, (batch_size,), replace_within_batch)

    keys = random.split(key, batches)
    perms = vmap(choose_batch, (0, None))(keys, batch_size)
    return tree_map(lambda x: x[perms], X)


def make_batches(data: PyTree, batch_size: int) -> BatchFn:
    def _make_batches(key: random.KeyArray, state: State):
        return tree_map(lambda X: batches_without_replacement(key, X, batch_size), data)

    return _make_batches


def train_model(
    key: random.KeyArray,
    state: State,
    train_step: TrainStep,
    epochs: int,
    batch_fn: BatchFn,
    compile: bool = False,
) -> tuple[State, PyTree]:
    def init_metric(m):
        arr = zeros((epochs,))
        arr = arr.at[:].set(jnp.nan)
        return arr.at[0].set(mean(m))

    key, subkey = random.split(key)
    state, aux_init = run_epoch(state, train_step, batch_fn(subkey, state))
    aux_init = tree_map(init_metric, aux_init)

    def body(loop_state):
        epoch, state, hist, key = loop_state
        key, subkey = random.split(key)
        batches = batch_fn(subkey, state)
        state, aux = run_epoch(state, train_step, batches)
        aux = tree_map(lambda v: mean(v), aux)
        hist = tree_map(lambda m, a: m.at[epoch].set(a), hist, aux)
        loss, metric = aux
        if hasattr(state, "on_epoch_end"):
            state = state.on_epoch_end(loss, metric)
        return epoch + 1, state, hist, key

    def cond(state):
        epoch, state, aux, _ = state
        loss, metric = aux
        epochs_complete = epoch >= epochs
        epoch_complete = epochs_complete

        if hasattr(state, "stop_training"):
            stop_training = state.stop_training(loss, metric)
        else:
            stop_training = False
        return ~(epoch_complete | stop_training)

    if compile:
        finished_epochs, state, aux, _ = lax.while_loop(
            cond, body, (1, state, aux_init, key)
        )
        _hist = aux
    else:
        train_state = (1, state, aux_init, key)
        while cond(train_state):
            train_state = body(train_state)
        finished_epochs, state, aux, _ = train_state
        _hist = tree_map(lambda a: a[:finished_epochs], aux)
    return state, _hist


# @dataclasses.dataclass(eq=False)
# class EarlyStopping(Callback):
#     validation_loss: Callable[[Params], Array]
#     patience: int = 0
#     mode: str = "min"
#     val_hist: list[Array] = dataclasses.field(repr=False, default_factory=list)
#     train_hist: list[Metric] = dataclasses.field(repr=False, default_factory=list)
#     best_state: None | State = dataclasses.field(repr=False, default=None)
#     counter: int = dataclasses.field(repr=False, default=0)

#     def __post_init__(self):
#         assert self.mode in ["min", "max"], "`mode` must be `min` or `max`"

#     def stop_training(self, state: State, train_metric: Metric) -> bool:
#         val_loss = self.validation_loss(state.params)
#         hist = stack(self.val_hist)
#         _stop_training = False
#         best_value = jnp.min(hist) if self.mode == "min" else jnp.max(hist)
#         op = jnp.less if self.mode == "min" else jnp.greater
#         if op(val_loss, best_value):
#             self.counter = 0
#             self.best_state = state
#         _stop_training = self.counter <= self.patience
#         if not _stop_training:
#             self.train_hist.append(train_metric)
#             self.val_hist.append(val_loss)
#             self.counter += 1
#         return _stop_training


class AdaptableScaleState(T.NamedTuple):
    scale: chex.Array


def adaptable_scale(init_scale: float) -> optax.GradientTransformation:
    def init_fn(params):
        del params
        return AdaptableScaleState(array(init_scale))

    def update_fn(updates, state, params=None):
        del params
        updates = jax.tree_util.tree_map(lambda g: state.scale * g, updates)
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)


@dataclass
class AdaptableLrTrainState(train_state.TrainState):
    patience: int = field(default=0)
    min_scale: float = field(default=1e-5)
    decrease_factor: float = field(default=0.5)
    stop_training_on_lr_min: bool = field(default=False)
    mode: str = field(pytree_node=False, default="min")
    best: Array = field(repr=False, default_factory=lambda: array(jnp.inf))
    counter: int = dataclasses.field(repr=False, default=0)

    def __post_init__(self):
        assert self.mode in ["min", "max"], "`mode` must be `min` or `max`"

    @classmethod
    def create(cls, *, apply_fn, params, tx, patience=0, mode="min", **kwargs):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        tx = optax.chain(tx, adaptable_scale(1.0))
        opt_state = tx.init(params)
        best = jnp.inf if mode == "min" else -jnp.inf
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            patience=patience,
            mode=mode,
            best=array(best),
            **kwargs,
        )

    @property
    def scale(self) -> Array:
        scale_state = self.opt_state[1]
        assert isinstance(scale_state, AdaptableScaleState)
        return scale_state.scale

    def reduce_lr(self) -> "AdaptableLrTrainState":
        opt_state = (
            self.opt_state[0],
            AdaptableScaleState(self.scale * self.decrease_factor),
        )
        return self.replace(opt_state=opt_state)

    def on_epoch_end(self, loss: Array, metric: Metric) -> "AdaptableLrTrainState":
        loss = loss if self.mode == "min" else -loss

        def on_improvement():
            return self.replace(best=loss, counter=0)

        def on_fail():
            return lax.cond(
                self.counter >= self.patience,
                lambda: self.reduce_lr().replace(counter=0),
                lambda: self.replace(counter=self.counter + 1),
            )

        return lax.cond(loss < self.best, on_improvement, on_fail)

    def stop_training(self, loss: Array, metric: Metric) -> Array:
        return array(self.stop_training_on_lr_min) & (self.scale <= self.min_scale)
