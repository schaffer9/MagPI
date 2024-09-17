from typing import Callable, Sequence, ParamSpec, Iterator

import chex

from magpi.prelude import *
from magpi.tr import TR


__all__ = (
    "TR",
    "batches_without_replacement",
    "batches_with_replacement",
    "make_iterator",
)


P = ParamSpec("P")
PyTree = chex.ArrayTree
Batch = PyTree
BatchFn = Callable[[Array], Sequence[Batch]]


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
    key: Array, X: PyTree, batch_size: int
) -> Sequence[Batch]:
    N = axis_size(X)
    batches = N // batch_size
    perms = jax.random.permutation(key, N)
    perms = perms[: batches * batch_size]
    perms = perms.reshape((batches, batch_size))
    return tree_map(lambda x: x[perms], X)


def batches_with_replacement(
    key: Array,
    X: PyTree,
    batches: int,
    batch_size: int,
    replace_within_batch=True,
) -> Sequence[Batch]:
    N = axis_size(X)

    def choose_batch(key, batch_size):
        return random.choice(key, N, (batch_size,), replace_within_batch)

    keys = random.split(key, batches)
    perms = vmap(choose_batch, (0, None))(keys, batch_size)
    return tree_map(lambda x: x[perms], X)


def make_iterator(
    key: Array, epochs: int, batch_fn: BatchFn, add_rng: bool = False
) -> Iterator[Batch]:
    for _ in range(epochs):
        key, sample_key, batch_key = random.split(key, 3)
        batches = batch_fn(sample_key)
        num_batches = axis_size(batches, 0)
        for i in range(num_batches):
            batch = tree_map(lambda b: b[i], batches)
            if add_rng:
                batch_key, subkey = random.split(batch_key)
                yield subkey, batch
            else:
                yield batch
