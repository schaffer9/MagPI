from pinns import domain

from . import *


class TestTransforms(JaxTestCase):
    def test_00_transform_hypercube(self):
        key = random.PRNGKey(0)
        x = random.uniform(key, (100, 10, 4))
        lb = array([-1, -1, -1, -1.])
        ub = array([1, 1, 1, 1.])

        x_tran = jit(domain.transform_hypercube)(x, lb, ub)
        self.assertTrue(((-1. <= x_tran) & (x_tran <= 1.)).all())
        self.assertTrue((x_tran < 0).any())
        self.assertEqual(x_tran.shape, (100, 10, 4))

    def test_01_transform_circle(self):
        key = random.PRNGKey(0)
        x = random.uniform(key, (100, 10, 2))
        r = array(2.)
        o = array([0., 0.])

        x_tran = domain.transform_circle(x, r, o)
        self.assertTrue((norm(x_tran, axis=-1) <= 2.).all())
        self.assertEqual(x_tran.shape, (100, 10, 2))

    def test_02_transform_circle(self):
        key = random.PRNGKey(0)
        x = random.uniform(key, (2,))
        r = array(2.)
        o = array([0., 1.])

        x_tran = jit(domain.transform_circle)(x, r, o)
        self.assertTrue((norm(x_tran - o, axis=-1) <= 2.).all())
        self.assertEqual(x_tran.shape, (2,))

    def test_03_transform_sphere(self):
        key = random.PRNGKey(0)
        x = random.uniform(key, (100, 3))
        r = array(0.1)
        o = array([0., 0., 1.])

        x_tran = jit(domain.transform_sphere)(x, r, o)
        self.assertTrue((norm(x_tran - o, axis=-1) <= 0.1).all())
        self.assertEqual(x_tran.shape, (100, 3))

    def test_04_transform_sphere_bnd(self):
        key = random.PRNGKey(0)
        x = random.uniform(key, (100, 10, 2))
        r = array(0.1)
        o = array([0., 0., 1.])

        x_tran = jit(domain.transform_sphere_bnd)(x, r, o)
        self.assertIsclose(norm(x_tran - o, axis=-1), 0.1)
        self.assertEqual(x_tran.shape, (100, 10, 3))

    def test_05_transform_circ_bnd(self):
        key = random.PRNGKey(0)
        x = random.uniform(key, (100, 10, 1))
        r = array(5.)
        o = array([0., 1.])

        x_tran = domain.transform_circle_bnd(x, r, o)
        self.assertIsclose(norm(x_tran - o, axis=-1), 5.)
        self.assertEqual(x_tran.shape, (100, 10, 2))

    def test_06_transform_circ_bnd_with_single_angle(self):
        key = random.PRNGKey(0)
        x = random.uniform(key)
        r = array(5.)
        o = array([0., 1.])

        x_tran = jit(domain.transform_circle_bnd)(x, r, o)
        self.assertIsclose(norm(x_tran - o, axis=-1), 5.)
        self.assertEqual(x_tran.shape, (2,))

    def test_07_hypercube_with_scalar(self):
        key = random.PRNGKey(0)
        x = random.uniform(key)
        lb = array(-1.)
        ub = array(1.)

        x_tran = jit(domain.transform_hypercube)(x, lb, ub)
        self.assertTrue(((-1. <= x_tran) & (x_tran <= 1.)).all())
        self.assertEqual(x_tran.shape, ())

    def test_08_transform_triangle(self):
        key = random.PRNGKey(0)
        x = random.uniform(key, (100, 10, 2))
        a, b, c = (0., 0.), (2., 0), (0., 2.)

        x_tran = domain.transform_triangle(x, a, b, c)
        self.assertTrue((norm(x_tran, 1, -1) <= 2.).all())
        self.assertTrue((norm(x_tran, 1, -1) > 1.).any())
        self.assertEqual(x_tran.shape, (100, 10, 2))

    def test_09_transform_polygon(self):
        x = array(0.)
        a, b, c = (0., 0.), (2., 0), (0., 2.)
        x_tran = domain.transform_polyline(x, (a, b, c))
        print(x_tran)
        self.assertIsclose(x_tran, array([0., 0.]))
        self.assertTrue(x_tran.shape, (2, ))

    def test_10_transform_polygon(self):
        x = array([0., 0.5, 1.])
        a, b, c = (0., 0.), (2., 0), (2., 2.)
        x_tran = domain.transform_polyline(x, (a, b, c))
        self.assertIsclose(x_tran, array([[0., 0.], [2, 0.], [0., 0.]]))
        self.assertTrue(x_tran.shape, (3, 2))

    def test_11_transform_hypercube_bnd(self):
        lb = array([-1, -1.])
        ub = array([1, 1.])
        x = array([[0], [0.75], [0.25]])
        x_trans = domain.transform_hypercube_bnd(x, lb, ub)
        self.assertIsclose(x_trans, array([[-1, -1], [1, -1.], [-1, -1.]]))

    def test_12_transform_hypercube_bnd(self):
        lb = array([-1, -2., -3.])
        ub = array([1, 2., 3.])
        key = random.PRNGKey(0)
        x = random.uniform(key, (100, 10, 2))
        x_trans = domain.transform_hypercube_bnd(x, lb, ub)
        self.assertEqual(x_trans.shape, (100, 10, 3))
        is_on_bnd = (x_trans == lb) | (x_trans == ub)
        # all points are on the edge
        self.assertTrue(is_on_bnd.any(axis=-1).all())
        

    def test_13_transform_hypercube_bnd(self):
        lb = array([-1, -2., -3., -2])
        ub = array([1, 2., 3., 2])
        key = random.PRNGKey(0)
        x = random.uniform(key, (100, 3))
        x_trans = domain.transform_hypercube_bnd(x, lb, ub)
        self.assertEqual(x_trans.shape, (100, 4))
        is_on_bnd = (x_trans == lb) | (x_trans == ub)
        self.assertTrue(is_on_bnd.any(axis=-1).all())
        # every edge has at least one sample mapped to it:
        self.assertTrue(is_on_bnd.any(axis=0).all())

    def test_14_transform_hypercube_bnd_one_dim(self):
        lb = array([-1.])
        ub = array([1.])
        key = random.PRNGKey(0)
        x = random.uniform(key, (10, 10))
        x_trans = domain.transform_hypercube_bnd(x, lb, ub)
        self.assertEqual(x_trans.shape, (10, 10))
        is_on_bnd = (x_trans == lb) | (x_trans == ub)
        self.assertTrue(is_on_bnd.any(axis=-1).all())
        

class TestDomain(JaxTestCase):
    def test_00_hypercube_init(self):
        cube = domain.Hypercube((1,), (2,))
        self.assertEqual(cube.dimension, 1)
        self.assertEqual(cube.lb, (1,))
        self.assertEqual(cube.ub, (2,))
        
    def test_01_hypercube_hash(self):
        cube1 = domain.Hypercube((1,), (2,))
        cube2 = domain.Hypercube((1,), (2,))
        self.assertEqual(hash(cube1), hash(cube2))

    def test_02_hypercube_domain_protocol(self):
        cube = domain.Hypercube((0., 0.), (1., 1.))
        self.assertEqual(cube.support(), 1.)
        x = array([0.5, 0.3])
        self.assertTrue(cube.includes(x))
        self.assertIsclose(cube.transform(x), x)

    def test_03_transform(self):
        cube = domain.Hypercube((0., 0.), (1., 1.))
        key = random.PRNGKey(0)
        x = random.uniform(key, (10, 2))
        x_tran = cube.transform(x)
        x_bnd = cube.transform_bnd(x[:, 0])
        self.assertEqual(x_tran.shape, (10, 2))
        self.assertTrue(cube.includes(x_tran))
        self.assertTrue(cube.includes(x_bnd))
    
    def test_04_normal_vec(self):
        cube = domain.Hypercube((0., 0., -1), (1., 1., 2))
        key = random.PRNGKey(0)
        x = random.uniform(key, (20, 2))
        x_bnd = cube.transform_bnd(x)
        n = cube.normal_vec(x_bnd)
        self.assertIsNotNone(n)
        self.assertIsclose(norm(n, axis=-1), 1.)

    def test_05_normal_vec_on_corner_is_defined(self):
        cube = domain.Hypercube((0., 0., -1), (1., 1., 2))
        n = cube.normal_vec(array([0, 0, -1.]))
        self.assertIsNotNone(n)
        self.assertIsclose(norm(n, axis=-1), 1.)
        self.assertIsclose(n, array([-1, 0, 0]))

    def test_06_sample_not_on_bnd_returns_none(self):
        cube = domain.Hypercube((0., 0., -1), (1., 1., 2))
        n = cube.normal_vec(array([-1,0,-1.]))
        self.assertIsNone(n)
        

        
    
