from magpi.integrate import integrate, gauss3, midpoint

from . import *


class TestIntegrate(JaxTestCase):
    def test_000_integrate_sin(self):
        f = lambda x: sin(x)
        d = linspace(0, pi, 30)
        F = integrate(f, d)
        self.assertEqual(F.shape, ())
        self.assertTrue(jnp.isclose(F, 2))
        
    def test_001_multioutput(self):
        f = lambda x: array(
            [[sin(x[0]), sin(x[1])], 
             [cos(x[1]), cos(x[0])]]
        )
        d = [
            linspace(0, 2 * pi, 100),
            linspace(0, 2 * pi, 200),
        ]
        F = integrate(f, d, method=gauss3)
        self.assertTrue(jnp.all(jnp.isclose(F, zeros((2, 2)), atol=1e-05)))

    def test_002_integrate_scalar_fun_with_midpoint(self):
        f = lambda x: 2 * x
        d = array([0.0, 1.0])
        F = integrate(f, d, method=midpoint)
        self.assertEqual(F.shape, ())
        self.assertIsclose(F, 1.0)