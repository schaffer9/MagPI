from magpi.spline import basis, SplineActivation

from . import *


class TestSplineBasis(JaxTestCase):
    def test_000_basis0(self):
        t = array([-1.0, -0.5, 0.0, 0.5, 1.0])
        x = array([-1.1, -1.0, -0.5, -0.1, 0.25, 1.0, 1.1])
        b = basis(x, t, 0, open_spline=True)
        self.assertIsclose(
            b,
            array(
                [
                    [0, 0, 0, 0],
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ]
            ),
        )

    def test_001_basis1(self):
        t = array([-1.0, -0.5, 0.0, 0.5, 1.0])
        x = -1
        b = basis(x, t, 1, open_spline=True)
        self.assertIsclose(b, array([0, 0, 0]))
        x = -2
        b = basis(x, t, 1, open_spline=True)
        self.assertIsclose(b, array([0, 0, 0]))
        x = 0
        b = basis(x, t, 1, open_spline=True)
        self.assertIsclose(b, array([0, 1, 0]))
        x = 1.0
        b = basis(x, t, 1, open_spline=True)
        self.assertIsclose(b, array([0, 0, 0]))
        x = 2.0
        b = basis(x, t, 1, open_spline=True)
        self.assertIsclose(b, array([0, 0, 0]))

    def test_002_basis2(self):
        t = array([-1.0, -0.5, 0.0, 0.5, 1.0])
        x = array([-1.0, -0.25, 0.25, 1.0])
        b = basis(x, t, 2, open_spline=True)
        self.assertIsclose(
            b,
            array(
                [
                    [0.0, 0.0],
                    [0.75, 0.125],
                    [0.125, 0.75],
                    [0.0, 0.0],
                ]
            ),
        )

    def test_003_vectorized(self):
        t = array([-1.0, -0.5, 0.0, 0.5, 1.0])
        t = asarray([t, t])
        x = array([-1.0, -0.25, 0.25, 1.0])
        x = asarray([x, x]).T
        b = basis(x, t, 2, open_spline=True)
        self.assertIsclose(
            b[:, 0],
            array(
                [
                    [0.0, 0.0],
                    [0.75, 0.125],
                    [0.125, 0.75],
                    [0.0, 0.0],
                ]
            ),
        )
        self.assertIsclose(
            b[:, 1],
            array(
                [
                    [0.0, 0.0],
                    [0.75, 0.125],
                    [0.125, 0.75],
                    [0.0, 0.0],
                ]
            ),
        )

    def test_004_closed_spline(self):
        t = array([-1.0, -0.5, 0.0, 0.5, 1.0])
        x = array([-1.1, -1.0, -0.25, 0.25, 1.0, 1.1])
        b = basis(x, t, 2, open_spline=False)
        result = array(
            [
                [1.44, -0.46000013, 0.02000001, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.125, 0.75, 0.125, 0.0, 0.0],
                [0.0, 0.0, 0.125, 0.75, 0.125, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.02000001, -0.46000013, 1.44],
            ]
        )
        self.assertIsclose(b, result)


class TestSplineActivation(JaxTestCase):
    def test_000_init_activation(self):
        c = array([1, 1])
        init = nn.initializers.constant(c)
        act = SplineActivation(
            nodes=5,
            degree=2,
            grid_power=1,
            node_min=0,
            node_max=1,
            activation=lambda x: 0.0,
            coef_init=init,
        )
        p = act.init(random.key(0), zeros((5,)))
        y = act.apply(p, ones((10, 5)))
        self.assertIsclose(y, zeros((10, 5)))
        x = random.uniform(random.key(0), (5, 5))
        y = act.apply(p, x)
        self.assertTrue(jnp.all(y > 0.0))

    def test_001_parameterize_grid(self):
        c = array([1, 1])
        init = nn.initializers.constant(c)
        act = SplineActivation(
            nodes=5,
            degree=2,
            grid_power=1,
            node_min=0,
            node_max=1,
            activation=lambda x: 0.0,
            coef_init=init,
            parameterize_grid=True
        )
        p = act.init(random.key(0), zeros((4,)))
        y = act.apply(p, ones((10, 4)))
        self.assertIsclose(y, zeros((10, 4)))
        x = random.uniform(random.key(0), (5, 4))
        y = act.apply(p, x)
        self.assertEqual(y.shape, (5, 4))
        self.assertTrue(jnp.all(y > 0.0))
        
        self.assertEqual(p["grids"]["grid"].shape, (4, 5))
