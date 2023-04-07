from pinns import sampling, domain

from . import *


class TestSampling(JaxTestCase):
    def test_01_rejection_sampling_withing_circle(self):
        pdf = lambda x: lax.cond(norm(x) <= 1., lambda: 1. / pi, lambda: 0.)
        sample_fn = lambda key: random.uniform(key, (2, )) * 2 - 1
        samples = sampling.rejection_sampling(
            random.PRNGKey(42),
            pdf, 
            sample_fn, 
            1024,
            4
        )
        self.assertTrue(all(norm(samples, axis=-1) <= 1.))

    def test_02_rejection_sample_as_pytree(self):
        _norm = lambda x: sqrt(x[0] ** 2 + x[1] ** 2)
        pdf = lambda x: lax.cond(_norm(x) <= 1., lambda: 1. / pi, lambda: 0.)
        
        def sample_fn(key):
            sample = random.uniform(key, (2, )) * 2 - 1
            return sample[0], sample[1]
        
        samples = sampling.rejection_sampling(
            random.PRNGKey(42),
            pdf, 
            sample_fn, 
            1024,
            4
        )
        self.assertTrue(all(vmap(_norm)(samples) <= 1.))
