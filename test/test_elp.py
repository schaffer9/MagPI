from magpi import integrate
from magpi import r_fun
from magpi.elp import (
    partition_domain, 
    legendre_polynomial, 
    legendre_poly_antiderivative, 
    center,
    _domain_grid,
    compute_elp,
    ELP,
    make_elp_quad_rule
)

from . import *


class TestLegendrePoly(JaxTestCase):
    def test_000_legendre_polynomial(self):
        d = array([-1, 1])
        L = integrate.integrate(legendre_polynomial, d, 5, method=integrate.gauss(3))
        result = array([2, 0, 0, 0, 0])
        self.assertIsclose(L, result)
        
    def test_001_integrate_2s(self):
        d = array([-1, 1])
        
        def _legendre_poly2d(x, n):
            L = legendre_polynomial(x, n)
            return jnp.outer(L[0], L[1])
        
        L = integrate.integrate(_legendre_poly2d, [d, d], 5, method=integrate.gauss(3))
        result = zeros((5, 5)).at[0, 0].set(4)
        self.assertIsclose(L, result)
        
    def test_002_antiderivative(self):
        d = array([-1, 0])
        a, b = legendre_poly_antiderivative(d, 5)
        I = b - a
        I_true = integrate.integrate(legendre_polynomial, d, 5, method=integrate.gauss(3))
        self.assertIsclose(I, I_true)
        
    def test_003_shifted_legendre_polynomial(self):
        lb, ub = 0, 1

        def poly(x, n):
            return legendre_polynomial(center(x, lb, ub), n)

        d = array([lb, ub])
        L = integrate.integrate(poly, d, 5, method=integrate.gauss(3))
        result = array([1, 0, 0, 0, 0])
        self.assertIsclose(L, result)
        

class TestPartitionDomain(JaxTestCase):
    def test_000_partition_cube(self):
        adf = r_fun.cube(2, centering=True)
        d = array([-1, 0, 1])
        cells = partition_domain(adf, [d, d, d], max_cells=1)
        V = jnp.sum(jnp.prod(cells.upper_bound - cells.lower_bound, axis=-1))
        self.assertEqual(V, 8)
        
    def test_001_partition_sphere(self):
        adf = r_fun.sphere(1)
        d = jnp.linspace(-1, 1, 4)
        d = _domain_grid([d, d, d])
        cells = partition_domain(adf, d, max_cells=100_000, max_depth=5, split_mode="boundary")
        V = jnp.sum(jnp.prod(cells.upper_bound - cells.lower_bound, axis=-1))
        V_true = 4 / 3 * pi
        self.assertIsclose(V, V_true, atol=1e-3)
        
    def test_002_partition_sphere_with_center_splitting(self):
        adf = r_fun.sphere(1)
        d = jnp.linspace(-1, 1, 4)
        d = _domain_grid([d, d, d])
        cells = partition_domain(adf, d, max_cells=100_000, max_depth=5, split_mode="center")
        V = jnp.sum(jnp.prod(cells.upper_bound - cells.lower_bound, axis=-1))
        V_true = 4 / 3 * pi
        self.assertIsclose(V, V_true, atol=1e-2)
        
    # def test_003_partition_convex_grain(self):
    #     # TODO
    #     assert False


class TestElp(JaxTestCase):
    def test_001_create_elp(self):
        d = jnp.linspace(-1, 1, 3)
        dom = _domain_grid([d, d])
        coefs = ones((2, 2, 6, 6))
        elp = ELP(dom, coefs)
        
        # ELP is integrated over subdomains and since coefficients are 1
        # all but the zero order term vanish. Therefore the integral
        # over the ELP is the area of the domain.
        I = integrate.integrate(elp, [d, d], method=integrate.gauss(3))
        self.assertIsclose(I, 4)
        
    def test_001_integrate_sphere_with_elp(self):
        adf = r_fun.sphere(1.0)
        d = jnp.linspace(-1, 1, 4)
        dom = _domain_grid([d, d, d])
        elp = compute_elp(adf, dom, 5, max_cells=100_000, max_depth=6)
        
        W, X = make_elp_quad_rule(elp)
        
        f = lambda x: jnp.prod(cos(x))
        I = integrate.integrate_quad_rule(f, W, X)
        I_true = integrate.integrate_sphere(f, 1.0, zeros((3,)), 7, method=integrate.gauss(5))
        self.assertIsclose(I, I_true, atol=1e-3)
        
    def test_002_elp_in_2d_with_different_degrees(self):
        adf = r_fun.sphere(1.0)
        d = jnp.linspace(-1, 1, 5)
        dom = _domain_grid([d, d])
        elp = compute_elp(adf, dom, (5, 6), max_cells=20_000, max_depth=7)
        
        W, X = make_elp_quad_rule(elp)
        f = lambda x: jnp.prod(cos(x))
        I = integrate.integrate_quad_rule(f, W, X)
        I_true = integrate.integrate_disk(f, 1.0, zeros((2,)), 7, method=integrate.gauss(5))
        self.assertIsclose(I, I_true, atol=1e-3)