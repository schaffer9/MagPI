from unittest import TestCase

from pinns.prelude import *


class JaxTestCase(TestCase):
    def assertIsclose(self, a, b):
        self.assertTrue(jnp.isclose(a, b, atol=1e-6).all(), 
        f"Arrays a ({array(a).shape}) and b ({array(b).shape}) are not close.")
