from pinns.r_fun import (
    Rp2Fun, 
    cube, 
    cuboid, 
    sphere, 
    Rp4Fun, 
    rotate2d,
    rotate3d,
    reflect,
    difference
)

from . import *


class TestRFunc(JaxTestCase):
    def test_000_intersection(self):
        df1 = Rp2Fun(lambda x: 1 - x)
        df2 = Rp2Fun(lambda x: x)
        df = df1 & df2
        self.assertIsclose(df(0.), 0.)
        self.assertIsclose(df(1.), 0.)
        self.assertTrue(df(0.5) > 0.)
        self.assertIsclose(grad(df)(array(0.)), 1.)
        self.assertIsclose(grad(df)(array(1.)), -1.)

    def test_001_union(self):
        df1 = Rp2Fun(lambda x: x)
        df2 = Rp2Fun(lambda x: x - 0.5)
        df = df1 | df2        
        self.assertIsclose(df(0.), 0.)
        self.assertTrue(df(0.5) > 0.)
        self.assertTrue(df(1.) > 0.)
        self.assertIsclose(grad(df)(array(0.)), 1.)
        self.assertTrue(grad(df)(array(1.)) > 1.)

    def test_002_xor(self):
        df1 = Rp2Fun(lambda x: 1 - x) & Rp2Fun(lambda x: x)
        df2 = Rp2Fun(lambda x: 0.5 - x) & Rp2Fun(lambda x: x + 0.5)
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
        df = ~(Rp2Fun(lambda x: 1 - x) & Rp2Fun(lambda x: x))
        self.assertIsclose(df(0.), 0.)
        self.assertIsclose(df(1.), 0.)
        self.assertTrue(df(0.5) < 0.)
        self.assertIsclose(grad(df)(array(0.)), -1.)
        self.assertIsclose(grad(df)(array(1.)), 1.)

    def test_004_difference(self):
        df2 = (Rp2Fun(lambda x: 1 - x) & Rp2Fun(lambda x: x))
        df1 = (Rp2Fun(lambda x: 1 - x) & Rp2Fun(lambda x: x - 0.5))
        df = difference(df2, df1)
        print(grad(df)(array(0.)))
        self.assertIsclose(df(0.), 0.)
        self.assertIsclose(df(0.5), 0.)
        self.assertIsclose(grad(df)(array(0.)), 1.)
        self.assertIsclose(grad(df)(array(0.5)), -1.)

    def test_005_translate(self):
        df = Rp2Fun(lambda x: 1 - x) & Rp2Fun(lambda x: x)
        df = df.translate(1.)
        self.assertIsclose(df(1.), 0.)
        self.assertIsclose(df(2.), 0.)
        self.assertTrue(df(1.5) > 0.)
        self.assertIsclose(grad(df)(array(1.)), 1.)
        self.assertIsclose(grad(df)(array(2.)), -1.)

    def test_006_scale(self):
        df = Rp2Fun(lambda x: 1 - x) & Rp2Fun(lambda x: x)
        df = df.scale(2.)
        self.assertIsclose(df(0.), 0.)
        self.assertIsclose(df(2.), 0.)
        self.assertTrue(df(1.5) > 0.)
        self.assertIsclose(grad(df)(array(0.)), 1.)
        self.assertIsclose(grad(df)(array(2.)), -1.)

    def test_007_scale(self):
        @jit
        def f(x):
            return cuboid([1., 1.], r_func=Rp4Fun).scale([1.])(x)

        self.assertIsclose(f(array([0.5, 1])), 0.)
        self.assertIsclose(f(array([0.5, 0])), 0.)
        self.assertIsclose(f(array([0., 0.5])), 0.)
        self.assertIsclose(f(array([1., 0.5])), 0.)
        self.assertIsclose(norm(grad(f)(array([0.5, 1]))), 1)
        self.assertIsclose(norm(grad(f)(array([0.5, 0]))), 1.)
        self.assertIsclose(norm(grad(f)(array([0., 0.5]))), 1.)
        self.assertIsclose(norm(grad(f)(array([1., 0.5]))), 1.)
        
        # higher order normalization is preserved
        def normal_derivative(g):
            return lambda x: -grad(f)(x) @ grad(g)(x)

        ddf = normal_derivative(normal_derivative(f))
        self.assertIsclose(ddf(array([0.5, 1])), 0.)
        self.assertIsclose(ddf(array([0.5, 0])), 0.)
        self.assertIsclose(ddf(array([0., 0.5])), 0.)
        self.assertIsclose(ddf(array([1., 0.5])), 0.)

        dddf = normal_derivative(ddf)
        self.assertIsclose(dddf(array([0.5, 1])), 0.)
        self.assertIsclose(dddf(array([0.5, 0])), 0.)
        self.assertIsclose(dddf(array([0., 0.5])), 0.)
        self.assertIsclose(dddf(array([1., 0.5])), 0.)
        
    def test_008_rotate2d(self):
        f = cube(1., dim=2)
        f = rotate2d(f, pi / 2, (0.5, 0.5))

        self.assertIsclose(f(array([0.5, 1])), 0.)
        self.assertIsclose(f(array([0.5, 0])), 0.)
        self.assertIsclose(f(array([0., 0.5])), 0.)
        self.assertIsclose(f(array([1., 0.5])), 0.)
        self.assertIsclose(norm(grad(f)(array([0.5, 1]))), 1)
        self.assertIsclose(norm(grad(f)(array([0.5, 0]))), 1.)
        self.assertIsclose(norm(grad(f)(array([0., 0.5]))), 1.)
        self.assertIsclose(norm(grad(f)(array([1., 0.5]))), 1.)

    def test_009_rotate3d(self):
        f = cube(1., dim=3)
        f = rotate3d(f, (0., 0., pi / 2), o=(0.5, 0.5, 0.5))

        self.assertIsclose(f(array([1., 0.5, 0.5])), 0.)
        self.assertIsclose(f(array([0.5, 1, 0.5])), 0.)
        self.assertIsclose(f(array([0.5, 0.5, 1.])), 0.)
        self.assertIsclose(f(array([0., 0.5, 0.5])), 0.)
        self.assertIsclose(f(array([0.5, 0, 0.5])), 0.)
        self.assertIsclose(f(array([0.5, 0.5, 0.])), 0.)

        self.assertIsclose(norm(grad(f)(array([1., 0.5, 0.5]))), 1)
        self.assertIsclose(norm(grad(f)(array([0.5, 1, 0.5]))), 1)
        self.assertIsclose(norm(grad(f)(array([0.5, 0.5, 1.]))), 1)
        self.assertIsclose(norm(grad(f)(array([0., 0.5, 0.5]))), 1)
        self.assertIsclose(norm(grad(f)(array([0.5, 0, 0.5]))), 1)
        self.assertIsclose(norm(grad(f)(array([0.5, 0.5, 0.]))), 1)

    def test_010_reflect2d(self):
        f = cube(1., dim=2).translate([-1., 0.])
        f = reflect(f, (1., 0.))

        self.assertIsclose(f(array([0.5, 1])), 0.)
        self.assertIsclose(f(array([0.5, 0])), 0.)
        self.assertIsclose(f(array([0., 0.5])), 0.)
        self.assertIsclose(f(array([1., 0.5])), 0.)
        self.assertIsclose(norm(grad(f)(array([0.5, 1]))), 1)
        self.assertIsclose(norm(grad(f)(array([0.5, 0]))), 1.)
        self.assertIsclose(norm(grad(f)(array([0., 0.5]))), 1.)
        self.assertIsclose(norm(grad(f)(array([1., 0.5]))), 1.)

    def test_011_reflect_with_origin(self):
        f = cube(1., dim=2).translate([1., 1])
        f = reflect(f, (1., 1.), o=(1., 1.))
        
        self.assertIsclose(f(array([0.5, 1])), 0.)
        self.assertIsclose(f(array([0.5, 0])), 0.)
        self.assertIsclose(f(array([0., 0.5])), 0.)
        self.assertIsclose(f(array([1., 0.5])), 0.)
        self.assertIsclose(norm(grad(f)(array([0.5, 1]))), 1)
        self.assertIsclose(norm(grad(f)(array([0.5, 0]))), 1.)
        self.assertIsclose(norm(grad(f)(array([0., 0.5]))), 1.)
        self.assertIsclose(norm(grad(f)(array([1., 0.5]))), 1.)

    def test_012_rotate_raises_value_error(self):
        f = cube(1., dim=3)
        with self.assertRaises(ValueError):
            rotate3d(f, pi)

    def test_013_rotate_raises_value_error(self):
        f = cube(1., dim=3)
        with self.assertRaises(ValueError):
            rotate3d(f, (pi, pi, pi), (1., 1., 1.))

class TestCube(JaxTestCase):
    def test_000_cube1d(self):
        df = cube(1., centering=True, dim=1)
        self.assertIsclose(df(array(-0.5)), 0.)
        self.assertIsclose(df(array(0.5)), 0.)