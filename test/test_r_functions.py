from pinns.r_functions import Rp2Func, cube, cuboid, sphere

from . import *


class TestRFunc(JaxTestCase):
    def test_000_intersection(self):
        df1 = Rp2Func(lambda x: 1 - x)
        df2 = Rp2Func(lambda x: x)
        df = df1 & df2
        self.assertIsclose(df(0.), 0.)
        self.assertIsclose(df(1.), 0.)
        self.assertTrue(df(0.5) > 0.)
        self.assertIsclose(grad(df)(array(0.)), 1.)
        self.assertIsclose(grad(df)(array(1.)), -1.)

    def test_001_union(self):
        df1 = Rp2Func(lambda x: x)
        df2 = Rp2Func(lambda x: x - 0.5)
        df = df1 | df2        
        self.assertIsclose(df(0.), 0.)
        self.assertTrue(df(0.5) > 0.)
        self.assertTrue(df(1.) > 0.)
        self.assertIsclose(grad(df)(array(0.)), 1.)
        self.assertTrue(grad(df)(array(1.)) > 1.)

    def test_002_xor(self):
        df1 = Rp2Func(lambda x: 1 - x) & Rp2Func(lambda x: x)
        df2 = Rp2Func(lambda x: 0.5 - x) & Rp2Func(lambda x: x + 0.5)
        df = df1 ^ df2
        print(grad(df)(0.5))
        self.assertIsclose(df(-0.5), 0.)
        self.assertIsclose(df(0.), 0.)
        self.assertIsclose(df(0.5), 0.)
        self.assertIsclose(df(1.), 0.)
        self.assertIsclose(grad(df)(-0.5), 1.)
        self.assertIsclose(grad(df)(0.5), 1.)
        self.assertIsclose(grad(df)(0.), -1.)
        self.assertIsclose(grad(df)(1.), -1.)

    def test_003_invert(self):
        df = ~(Rp2Func(lambda x: 1 - x) & Rp2Func(lambda x: x))
        self.assertIsclose(df(0.), 0.)
        self.assertIsclose(df(1.), 0.)
        self.assertTrue(df(0.5) < 0.)
        self.assertIsclose(grad(df)(array(0.)), -1.)
        self.assertIsclose(grad(df)(array(1.)), 1.)

    def test_004_div(self):
        df2 = (Rp2Func(lambda x: 1 - x) & Rp2Func(lambda x: x))
        df1 = (Rp2Func(lambda x: 1 - x) & Rp2Func(lambda x: x - 0.5))
        df = df2 / df1
        print(grad(df)(array(0.)))
        self.assertIsclose(df(0.), 0.)
        self.assertIsclose(df(0.5), 0.)
        self.assertIsclose(grad(df)(array(0.)), 1.)
        self.assertIsclose(grad(df)(array(0.5)), -1.)

    def test_005_translate(self):
        df = Rp2Func(lambda x: 1 - x) & Rp2Func(lambda x: x)
        df = df.translate(1.)
        self.assertIsclose(df(1.), 0.)
        self.assertIsclose(df(2.), 0.)
        self.assertTrue(df(1.5) > 0.)
        self.assertIsclose(grad(df)(array(1.)), 1.)
        self.assertIsclose(grad(df)(array(2.)), -1.)

    def test_006_scale(self):
        df = Rp2Func(lambda x: 1 - x) & Rp2Func(lambda x: x)
        df = df.scale(2.)
        self.assertIsclose(df(0.), 0.)
        self.assertIsclose(df(2.), 0.)
        self.assertTrue(df(1.5) > 0.)
        self.assertIsclose(grad(df)(array(0.)), 1.)
        self.assertIsclose(grad(df)(array(2.)), -1.)

    def test_007_scale(self):
        @jit
        def df(x):
            df = cuboid([1., 1.]).scale([1., 2])
            return df(x)

        self.assertIsclose(df(array([0.5, 2])), 0.)
        self.assertIsclose(df(array([0.5, 0])), 0.)
        self.assertIsclose(df(array([0., 1.])), 0.)
        self.assertIsclose(df(array([1., 1.])), 0.)
        self.assertIsclose(norm(grad(df)(array([0.5, 2]))), 1)
        self.assertIsclose(norm(grad(df)(array([0.5, 0]))), 1.)
        self.assertIsclose(norm(grad(df)(array([0., 1.]))), 1.)
        self.assertIsclose(norm(grad(df)(array([1., 1.]))), 1.)


class TestCube(JaxTestCase):
    def test_000_cube1d(self):
        df = cube(1., centering=True, dim=1)
        self.assertIsclose(df(array(-0.5)), 0.)
        self.assertIsclose(df(array(0.5)), 0.)