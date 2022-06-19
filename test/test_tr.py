from pinns import tr
from test import JaxTestCase

from pinns import tr
from pinns import calc

from . import *


class TestTR(JaxTestCase):
    def test_000_steihaug(self):
        x0 = array(2.)
        df = 2 * x0
        hvp = lambda x: 2 * x
        iterations, p = tr.steihaug(df, hvp, delta=2., maxiter=10, eps=1e-7)
        self.assertIsclose(x0 + p, 0.)
        self.assertIsclose(abs(p), 2.)
        self.assertEqual(iterations, 1)

    def test_001_steihaug_limit_step(self):
        x0 = array(2.)
        df = 2 * x0
        hvp = lambda x: 2 * x
        iterations, p = tr.steihaug(df, hvp, delta=1., maxiter=10, eps=1e-7)
        self.assertIsclose(x0 + p, 1.)
        self.assertIsclose(abs(p), 1.)
        self.assertEqual(iterations, 1)

    def test_002_steihaug_eps_reached(self):
        x0 = array(2.)
        df = 2 * x0
        hvp = lambda x: 2 * x
        iterations, p = tr.steihaug(df, hvp, delta=1., maxiter=10, eps=5)
        self.assertIsclose(x0 + p, 2.)
        self.assertIsclose(abs(p), 0.)
        self.assertEqual(iterations, 0)

    def test_003_steihaug_2d(self):
        x0 = array([1., 1.])
        f = lambda x: array([0., 0.]) @ x + x @ array([[1., 0], [0., 1.]]) @ x
        df = grad(f)(x0)
        hvp = lambda x: calc.hvp(f, (x0,), (x,))
        iterations, p = jit(tr.steihaug, static_argnames="hvp")(df, hvp, delta=3, maxiter=2, eps=1e-7)
        self.assertIsclose(x0 + p, array([0., 0.]))
        self.assertIsclose(p, array([-1., -1.]))
        self.assertEqual(iterations, 1)

    def test_004_tr_branin_function(self):
        f = lambda x: (x[1] - 0.129 * x[0] ** 2 + 1.6 * x[0] - 6) ** 2 + 6.07 * cos(x[0]) + 10
        x0 = array([6., 14.], dtype=jnp.float64)
        result = jit(tr.tr, static_argnames='f')(f, x0, delta_0 = 2.,
            delta_max=2., eps_grad=0.001, maxiter=30, eps_steihaug=1e-3)
        self.assertIsclose(result.params, array([3.1415486, 2.2468324]))
        self.assertEqual(result.delta, 2.)
        self.assertLess(result.grad_norm, 0.001)
        