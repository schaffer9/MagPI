from .prelude import *

from .domain import Domain

@partial(jit, static_argnames="domain")
def sample_domain(key, domain: Domain, n: int=1):
    assert n >= 1
    d = (domain.dimension, ) if n == 1 else (n, domain.dimension)
    uniform_sample = random.uniform(key, d)
    return domain.transform(uniform_sample)


@partial(jit, static_argnames=("condition", "sampling", "n"))
def rejection_sampling(key, condition, sampling, n: int):
    key, subkey = random.split(key)
    sample = sampling(subkey)
    shape = (n, *sample.shape)
    samples = jnp.empty(shape, dtype=sample.dtype)
    i = 0

    def body(state):
        samples, i, key, sample = state
        valid = condition(sample)
        key, subkey = random.split(key)
        samples = lax.cond(
            valid, 
            lambda samples: samples.at[i].set(sample), 
            lambda samples: samples,
            samples
        )
        i = lax.cond(valid, lambda i: i+1, lambda i: i, i)
        sample = sampling(subkey)
        return (samples, i, key, sample)

    samples, *_ = lax.while_loop(
        lambda state: state[1] < n,
        body,
        (samples, i, key, sample)
    )

    return samples