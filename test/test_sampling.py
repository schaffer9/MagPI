from pinns import sampling, domain

from . import *


class TestSampling(JaxTestCase):
    def test_00_sample_uniform(self):
        d = domain.Hypercube((0.,), (1.,))
        key = random.PRNGKey(42)
        k1, k2 = random.split(key)
        sample1 = sampling.sample_domain(k1, d)
        sample2 = sampling.sample_domain(k2, d)
        
        self.assertEqual(sample1.shape, (1, ))
        self.assertEqual(sample2.shape, (1, ))
        self.assertFalse(jnp.isclose(sample1, sample2).any())

    def test_01_rejection_sampling(self):
        d = domain.Hypercube((0., 0.), (1., 1.))
        key = random.PRNGKey(42)

        def condition(s, a):
            return (s[..., 0] > 0.5) & (s[..., 1] > 0.5)

        samples = sampling.rejection_sampling(
            key, 
            condition,
            lambda k: sampling.sample_domain(k, d),
            100, condition_kwargs={'a': 42}  # 42 is a dummy parameter
        )

        self.assertEqual(samples.shape, (100, 2))
        self.assertTrue((samples > 0.5).all())
        self.assertTrue((samples < 1.).all())

    