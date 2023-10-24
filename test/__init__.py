from unittest import TestCase

from magpi.prelude import *


class JaxTestCase(TestCase):
    def assertIsclose(self, a, b):
        self.assertTrue(
            tree_map(lambda a, b: jnp.isclose(a, b, atol=1e-6).all(), a, b),
            f"Arrays a ({array(a).shape}) and b ({array(b).shape}) are not close.",
        )
