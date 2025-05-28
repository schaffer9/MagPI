from jax.scipy import stats

from magpi import sampling
from magpi.calc import laplace
from magpi.magnetostatic import create_scalar_potential_solver
from magpi.r_fun import cube
from magpi.spline import basis
from magpi.integrate import make_quad_rule, gauss
from magpi.mesh import Mesh
from . import *


class TestSampling(JaxTestCase):
    def test_01_rejection_sampling_withing_circle(self):
        pdf = lambda x: lax.cond(norm(x) <= 1., lambda: 1. / pi, lambda: 0.)
        
        def accept_fn(sample, key):
            p = random.uniform(key)
            return p < (pdf(sample) / 4)
        
        sample_fn = lambda key: random.uniform(key, (2, )) * 2 - 1
        samples = sampling.rejection_sampling(random.PRNGKey(42), 1024, sample_fn, accept_fn)
        self.assertTrue(all(norm(samples, axis=-1) <= 1.))

    def test_02_rejection_sample_as_pytree(self):
        _norm = lambda x: sqrt(x[0] ** 2 + x[1] ** 2)
        pdf = lambda x: lax.cond(_norm(x) <= 1., lambda: 1. / pi, lambda: 0.)
        
        def accept_fn(sample, key):
            p = random.uniform(key)
            return p < (pdf(sample) / 4)
        
        def sample_fn(key):
            sample = random.uniform(key, (2, )) * 2 - 1
            return sample[0], sample[1]
        
        samples = sampling.rejection_sampling(random.PRNGKey(42), 1024, sample_fn, accept_fn)
        self.assertTrue(all(vmap(_norm)(samples) <= 1.))


def h_elm(x):
    degree = 4
    n = 6
    t = jnp.linspace(-1, 1, n)
    h0 = basis(x[0], t, degree=degree)
    h1 = basis(x[1], t, degree=degree)
    h2 = basis(x[2], t, degree=degree)
    return jnp.outer(jnp.outer(h0, h1), h2).ravel()


class TestSampleMagnetizationStates(JaxTestCase):
    def test_001_sample_mags(self):
        adf = cube(1, centering=True)
        cube_domain = [jnp.linspace(-0.5, 0.5, 6)] * 3
        W, X = make_quad_rule(cube_domain, method=gauss(5), ravel=True)
        quad_rule = (W, X)
        tri_quad_rule = (asarray([0.5]), asarray([[1 / 3, 1 / 3]]))
        
        solver = create_scalar_potential_solver(
            adf, h_elm, quad_rule, 
            Mesh(array([]), array([]), array([]), 0, 0.0), tri_quad_rule
        )
        
        key = random.key(42)
        tol = 5e-2
        mags, poisson_solution, _ = sampling.sample_magnetization_states(
            key, 10, solver, tol=tol
        )
        self.assertTrue(jnp.all(poisson_solution.strong_residual < tol))
        m0 = tree.map(lambda t: t[0], mags)
        m = sampling.default_mag_model(zeros((3,)), m0)
        self.assertEqual(m.shape, (3,))


class TestSampleDomain(JaxTestCase):
    def test_001_sample_cube(self):
        key = random.key(0)
        adf = cube(2, centering=True)
        samples = sampling.sample_domain(
            key, 1000, adf, dimension=3,
            lower_bound=-2, upper_bound=2, eps=1e-2, curvature_threshold=10,
            interior=True
        )
        l = vmap(adf)(samples)
        lap_l = vmap(laplace(adf))(samples)
        self.assertTrue(jnp.all(l > 1e-2))
        self.assertTrue(jnp.all(jnp.abs(lap_l) < 10))

    def test_002_sample_cube_exterior(self):
        key = random.key(0)
        mu = array([0., 0., 0.])
        cov = jnp.identity(3) * 5
        pdf = lambda x: stats.multivariate_normal.pdf(x, mean=mu, cov=cov)
        adf = cube(2, centering=True)
        samples = sampling.sample_domain(
            key, 1000, adf, dimension=3,
            lower_bound=-10, upper_bound=10, eps=1e-2, curvature_threshold=10,
            interior=False, pdf=pdf
        )
        l = vmap(adf)(samples)
        lap_l = vmap(laplace(adf))(samples)
        self.assertTrue(jnp.all(l < 1e-2))
        self.assertTrue(jnp.all(jnp.abs(lap_l) < 10))