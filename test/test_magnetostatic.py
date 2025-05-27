
import pytest
from scipy.stats.qmc import Sobol

from magpi.domain import Sphere
from magpi.r_fun import sphere, cube
from magpi.spline import basis
from magpi.magnetostatic import (
    create_elm_poisson_solver,
    create_slp_solver,
    create_scalar_potential_solver,
    create_vector_potential_solver,
    ScalarPotentialSolver,
    VectorPotentialSolver
)
from magpi.integrate import make_quad_rule, load_tri_quad_rule, gauss
from magpi.grain import generate_convex_mesh
from magpi.sampling import flower_state

from . import *


def h_elm(x):
    degree = 4
    n = 6
    t = jnp.linspace(-1, 1, n)
    h0 = basis(x[0], t, degree=degree)
    h1 = basis(x[1], t, degree=degree)
    h2 = basis(x[2], t, degree=degree)
    return jnp.outer(jnp.outer(h0, h1), h2).ravel()


class TestPoissonSolver(JaxTestCase):
    def test_001_solve_poisson_on_sphere(self):
        adf = sphere(1.0)
        X = asarray(Sobol(3, rng=42).random_base2(10))
        X = Sphere(1.0, (0.0, 0.0, 0.0)).transform(X)
        W = ones((X.shape[0],)) / X.shape[0]
        quad_rule = (W, X)
        
        solver = create_elm_poisson_solver(adf, h_elm, quad_rule)
        
        def rhs(x):
            return sin(x[0]) ** 2 * cos(x[1])
        
        elm_solution = solver.solve(rhs)
        self.assertTrue(elm_solution.strong_residual < 1e-1)
    
    def test_002_flatten_and_unflatten(self):
        adf = sphere(1.0)
        X = asarray(Sobol(3, rng=42).random_base2(10))
        X = Sphere(1.0, (0.0, 0.0, 0.0)).transform(X)
        W = ones((X.shape[0],)) / X.shape[0]
        quad_rule = (W, X)
        
        solver1 = create_elm_poisson_solver(adf, h_elm, quad_rule)
        leaves, treedef = tree.flatten(solver1)
        solver2 = tree.unflatten(treedef, leaves)
        
        self.assertTrue(jnp.all(solver1.Q == solver2.Q))
     
        
class TestSlpSolver(JaxTestCase):
    @pytest.mark.skipif(not ngsolve_installed(), reason="requires NGSolve")
    @pytest.mark.skipif(not has_internet(), reason="requires internet connection")
    def test_001_create_slp_solver(self):
        _box = array(
            [
                [-1.0, 0.0, 0.0, -0.5],
                [0.0, -1.0, 0.0, -0.5],
                [-0.0, -0.0, -1.0, -0.5],
                [0.0, 0.0, 1.0, -0.5],
                [0.0, 1.0, 0.0, -0.5],
                [1.0, 0.0, 0.0, -0.5],
            ]
        )
        tri_quad_rule = load_tri_quad_rule(30)
        mesh = generate_convex_mesh(_box, maxh=0.2).to_jax()
        solver = create_slp_solver(mesh, tri_quad_rule, 2)
        z, dz = solver.compute_source(zeros((3,)))
        self.assertEqual(z.shape, (mesh.sur_elements.shape[0], 13))
        self.assertEqual(dz.shape, (mesh.sur_elements.shape[0], 13, 3))
        

class TestScalarPotentialSolver(JaxTestCase):
    @pytest.mark.skipif(not ngsolve_installed(), reason="requires NGSolve")
    @pytest.mark.skipif(not has_internet(), reason="requires internet connection")
    def test_001_solve_flower_state(self):
        _box = array(
            [
                [-1.0, 0.0, 0.0, -0.5],
                [0.0, -1.0, 0.0, -0.5],
                [-0.0, -0.0, -1.0, -0.5],
                [0.0, 0.0, 1.0, -0.5],
                [0.0, 1.0, 0.0, -0.5],
                [1.0, 0.0, 0.0, -0.5],
            ]
        )
        adf = cube(1, centering=True)
        cube_domain = [jnp.linspace(-0.5, 0.5, 6)] * 3
        W, X = make_quad_rule(cube_domain, method=gauss(5), ravel=True)
        quad_rule = (W, X)
        tri_quad_rule = load_tri_quad_rule(30)
        mesh = generate_convex_mesh(_box, maxh=0.2)
        solver = create_scalar_potential_solver(
            adf, h_elm, quad_rule, mesh, tri_quad_rule, eps=1e-4, order=2
        )
        
        @jit
        def solve_flower_state(solver: ScalarPotentialSolver):
            X = solver.poisson_solver.quad_rule[1]
            Z, dZ = lax.map(solver.slp_solver.compute_source, X)
            sources = (X, Z, dZ)
            potential = solver.solve(flower_state, sources)
            return potential
        
        potential = solve_flower_state(solver)
        H = potential.field
        M = vmap(flower_state)(X)
        energy = - 1 / 2 * jnp.sum(W * jnp.sum(H * M, axis=-1))
        self.assertIsclose(energy, 0.1528, atol=1e-3)
        

class TestVectorPotentialSolver(JaxTestCase):
    @pytest.mark.skipif(not ngsolve_installed(), reason="requires NGSolve")
    @pytest.mark.skipif(not has_internet(), reason="requires internet connection")
    def test_001_solve_flower_state(self):
        _box = array(
            [
                [-1.0, 0.0, 0.0, -0.5],
                [0.0, -1.0, 0.0, -0.5],
                [-0.0, -0.0, -1.0, -0.5],
                [0.0, 0.0, 1.0, -0.5],
                [0.0, 1.0, 0.0, -0.5],
                [1.0, 0.0, 0.0, -0.5],
            ]
        )
        adf = cube(1, centering=True)
        cube_domain = [jnp.linspace(-0.5, 0.5, 6)] * 3
        W, X = make_quad_rule(cube_domain, method=gauss(5), ravel=True)
        quad_rule = (W, X)
        tri_quad_rule = load_tri_quad_rule(30)
        mesh = generate_convex_mesh(_box, maxh=0.2)
        solver = create_vector_potential_solver(
            adf, h_elm, quad_rule, mesh, tri_quad_rule, eps=1e-4, order=2
        )
        
        @jit
        def solve_flower_state(solver: VectorPotentialSolver):
            X = solver.poisson_solver.quad_rule[1]
            Z, dZ = lax.map(solver.slp_solver.compute_source, X)
            sources = (X, Z, dZ)
            potential = solver.solve(flower_state, sources)
            return potential
        
        potential = solve_flower_state(solver)
        B = potential.field
        M = vmap(flower_state)(X)
        energy = 1 / 2 * (1 - jnp.sum(W * jnp.sum(B * M, axis=-1)))
        self.assertIsclose(energy, 0.1528, atol=1e-3)