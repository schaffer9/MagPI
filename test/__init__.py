import socket
from unittest import TestCase

from magpi.prelude import *
from magpi.mesh import NGSOLVE_INSTALLED


class JaxTestCase(TestCase):
    def assertIsclose(self, a, b, rtol=1e-05, atol=1e-6):
        def _asarray(a):
            return tree_map(lambda a: asarray(a), a)
        
        a, b = _asarray(a), _asarray(b)
        shapes_are_the_same = all(tree_leaves(tree_map(lambda a, b: a.shape == b.shape, a, b)))
        
        def _shape(a):
            return tree_map(lambda a: a.shape, a)
        
        self.assertTrue(shapes_are_the_same, 
                        f"Arrays a ({_shape(a)}) and b ({_shape(b)}) don't have the same shape.")
        self.assertTrue(
            tree_map(lambda a, b: jnp.isclose(a, b, atol=atol, rtol=rtol).all(), a, b),
            f"Arrays a ({_shape(a)}) and b ({_shape(b)}) are not close.",
        )

    def assertPytreeEqual(self, a, b):
        a_shapes = tree.map(lambda a: asarray(a).shape, a)
        b_shapes = tree.map(lambda b: asarray(b).shape, b)
        self.assertEqual(a_shapes, b_shapes, f"Shapes of pytrees `{a_shapes}` and `{b_shapes}` are not equal")
        a_leaves = tree.leaves(a)
        b_leaves = tree.leaves(b)
        all_equal = all([jnp.all(_a == _b) for _a, _b in zip(a_leaves, b_leaves)])
        self.assertTrue(all_equal, "Arrays are not equal")


def has_internet(host="8.8.8.8", port=53, timeout=1.5):
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error:
        return False
    

def ngsolve_installed():
    return NGSOLVE_INSTALLED