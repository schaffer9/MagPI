from . import JaxTestCase
from magpi.prelude import *
from magpi import model
from magpi.r_fun import cube


class TestImposeNeumannBC(JaxTestCase):
    def test_00_homogenious_bc(self):
        def f(x):
            return x

        adf = cube(1.0)

        f_new = model.impose_neumann_bc(adf, f)
        _, normal_derivative = jvp(f_new, [0.0], [-1.0])
        self.assertIsclose(normal_derivative, 0.0)

    def test_01_2d(self):
        def f(x):
            return array([x, 2 * x])

        adf = cube(1.0)
        h = lambda x: array([1.0, 0])
        f_new = model.impose_neumann_bc(adf, f, h)
        _, normal_derivative = jvp(f_new, [array([0.0, 0.5])], [array([-1.0, 0.0])])
        self.assertIsclose(normal_derivative, array([1.0, 0]))


class TestImposeDirichletBC(JaxTestCase):
    def test_00_homogenious_bc(self):
        def f(x):
            return x

        adf = cube(1.0)
        f_new = model.impose_dirichlet_bc(adf, f)
        value = f_new(array([0.0, 0.5]))
        self.assertIsclose(value, array([0.0, 0]))

    def test_01_2d(self):
        def f(x):
            return array([cos(x), sin(x)])

        adf = cube(1.0)
        g = lambda x: array([1.0, 0])
        f_new = model.impose_dirichlet_bc(adf, f, g)
        value = f_new(array([0.0, 0.5]))
        self.assertIsclose(value, array([1.0, 0]))


class TestImposeIC(JaxTestCase):
    def test_00_impose_ic(self):
        def f(x, t):
            return 0.0

        f_new = model.impose_ic(
            lambda x: 1.0,
            f,
        )
        self.assertIsclose(f_new(20.0, 0.0), 1.0)
        self.assertIsclose(f_new(20.0, 1000.0), 0.0)
