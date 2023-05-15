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
        converged, iterations, p = tr.steihaug(df, hvp, tr_radius=2., maxiter=10)
        self.assertTrue(converged)
        self.assertIsclose(x0 + p, 0.)
        self.assertIsclose(abs(p), 2.)
        self.assertEqual(iterations, 1)

    def test_001_steihaug_limit_step(self):
        x0 = array(2.)
        df = 2 * x0
        hvp = lambda x: 2 * x
        converged, iterations, p = tr.steihaug(df, hvp, tr_radius=1., maxiter=10)
        self.assertTrue(converged)
        self.assertIsclose(x0 + p, 1.)
        self.assertIsclose(abs(p), 1.)
        self.assertEqual(iterations, 1)

    def test_002_steihaug_eps_reached(self):
        x0 = array(0.)
        df = 2 * x0
        hvp = lambda x: 2 * x
        converged, iterations, p = tr.steihaug(df, hvp, tr_radius=2., maxiter=10, eps_max=1e-4)
        self.assertTrue(converged)
        self.assertIsclose(x0 + p, 0.)
        self.assertIsclose(abs(p), 0.)
        self.assertEqual(iterations, 0)

    def test_003_steihaug_2d(self):
        x0 = (array([1., 1.]), )
        f = lambda x: array([0., 0.]) @ x[0] + x[0] @ array([[1., 0], [0., 1.]]) @ x[0]
        df = grad(f)(x0)
        #hvp = lambda x: calc.hvp(f, (x0,), (x,))
        hvp = lambda x: calc.hvp_forward_over_reverse(f, (x0,), (x,))
        steihaug = jit(tr.steihaug, static_argnames=("hvp", "maxiter"))
        converged, iterations, p = steihaug(df, hvp, tr_radius=3, maxiter=2)
        self.assertTrue(converged)
        self.assertIsclose(x0[0] + p[0], array([0., 0.]))
        self.assertIsclose(p[0], array([-1., -1.]))
        self.assertEqual(iterations, 1)

    def test_004_tr_branin_function(self):
        f = lambda x: (x[1] - 0.129 * x[0] ** 2 + 1.6 * x[0] - 6) ** 2 + 6.07 * cos(x[0]) + 10
        x0 = array([6., 14.])
        
        solver = tr.TR(
            f, 
            init_tr_radius=1.,
            max_tr_radius=2.,
            tol=1e-4,
            maxiter=30,
            eps_steihaug=1e-2,
            maxiter_steihaug=2
        )
        params, state = jit(solver.run)(x0)
        self.assertIsclose(params, array([3.1415927, 2.2466307]))
        self.assertEqual(state.tr_radius, 2.)
        self.assertLess(tree_l2_norm(state.grad), 0.001)
    
    def test_005_tr_with_pytree_params(self):
        f = lambda x: (x['params'][1] - 0.129 * x['params'][0] ** 2 + 1.6 * x['params'][0] - 6) ** 2 + 6.07 * cos(x['params'][0]) + 10
        x0 = {'params': array([6., 14.])}
        
        solver = tr.TR(
            f, 
            init_tr_radius=1.,
            max_tr_radius=2.,
            tol=1e-4,
            maxiter=30,
            eps_steihaug=1e-2,
            maxiter_steihaug=2
        )
        params, state = jit(solver.run)(x0)
        self.assertIsclose(params['params'], array([3.1415927, 2.2466307]))
        self.assertEqual(state.tr_radius, 2.)
        self.assertLess(tree_l2_norm(state.grad), 0.001)