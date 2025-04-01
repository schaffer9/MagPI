from magpi.spline import basis

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
