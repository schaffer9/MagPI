from magpi.integrate import integrate, gauss3

from . import *


class TestIntegrate(JaxTestCase):
    def test_000_integrate_sin(self):
        f = lambda x: sin(x)
        d = linspace(0, pi, 30)
        F = integrate(f, d)
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
