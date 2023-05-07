from .prelude import *
from chex import ArrayTree


Sample = ArrayTree
PDF = Callable[[Sample], float]
SampleFn = Callable[[random.KeyArray], Sample]


def rejection_sampling(
    key: random.KeyArray,
    pdf: PDF,
    sample_fn: SampleFn,
    n: int,
    m: int,
):
    """Draws `n` samples according to the given PDF. It takes on average
    `m` iterations for each sample. Samples are drawn in parallel.

    Parameters
    ----------
    key : random.KeyArray
    pdf : PDF
    sample_fn : SampleFn
    n : int
    m : int
    """

    def draw_sample(key):
        key, samplekey, valkey = random.split(key, 3)
        sample = sample_fn(samplekey)
        p = random.uniform(valkey)

        def body(state):
            key, p, sample = state
            key, samplekey, valkey = random.split(key, 3)
            p = random.uniform(valkey)
            sample = sample_fn(samplekey)
            return key, p, sample

        def not_valid(state):
            _, p, sample = state
            return p > (pdf(sample) / m)

        _, _, sample = lax.while_loop(not_valid, body, (key, p, sample))
        return sample

    keys = random.split(key, n)
    return vmap(draw_sample)(keys)
