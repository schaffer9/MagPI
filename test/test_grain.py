import pytest

import numpy as np
from magpi.grain import sample_grain, create_grain, shift_grain

from . import *


class TestGrain(JaxTestCase):
    @pytest.mark.skipif(not ngsolve_installed(), reason="requires NGSolve")
    def test_000_create_cubic_grain_with_surface_mesh(self):
        cube_eq = array(
            [
                [-1.0, 0.0, 0.0, -1.0],
                [0.0, -1.0, 0.0, -1.0],
                [-0.0, -0.0, -1.0, -1.0],
                [0.0, 0.0, 1.0, -1.0],
                [0.0, 1.0, 0.0, -1.0],
                [1.0, 0.0, 0.0, -1.0],
            ]
        )
        grain = create_grain(
            cube_eq,
            create_quad_rule=False,
            create_mesh=True,
            maxh=0.3,
            max_elements=200,
            surface_mesh=True,
        )

        self.assertIsclose(grain.equations, cube_eq)
        self.assertEqual(grain.mesh.sur_elements.shape[0], 200)
        self.assertEqual(grain.volume, 8)

    @pytest.mark.skipif(not ngsolve_installed(), reason="requires NGSolve")
    def test_001_create_cubic_grain_with_inactive_equation(self):
        cube_eq = array(
            [
                [-1.0, 0.0, 0.0, -1.0],
                [0.0, -1.0, 0.0, -1.0],
                [-0.0, -0.0, -1.0, -1.0],
                [0.0, 0.0, 1.0, -1.0],
                [0.0, 1.0, 0.0, -1.0],
                [1.0, 0.0, 0.0, -1.0],
                [1.0, 0.0, 0.0, -1.1],  # <- inactive
            ]
        )
        grain = create_grain(
            cube_eq,
            create_quad_rule=False,
            create_mesh=False,
        )
        result_eq = array(
            [
                [-1.0, 0.0, 0.0, -1.0],
                [0.0, -1.0, 0.0, -1.0],
                [-0.0, -0.0, -1.0, -1.0],
                [0.0, 0.0, 1.0, -1.0],
                [0.0, 1.0, 0.0, -1.0],
                [1.0, 0.0, 0.0, -1.0],
                [0.0, 0.0, 0.0, 0.0],  # <- turns to zero
            ]
        )

        self.assertEqual(grain.volume, 8)
        self.assertIsclose(grain.equations, result_eq)

    def test_002_create_cubic_grain_with_padding_eqation_and_jit(self):
        cube_eq = array(
            [
                [-1.0, 0.0, 0.0, -1.0],
                [0.0, -1.0, 0.0, -1.0],
                [-0.0, -0.0, -1.0, -1.0],
                [0.0, 0.0, 1.0, -1.0],
                [0.0, 1.0, 0.0, -1.0],
                [1.0, 0.0, 0.0, -1.0],
                [0.0, 0.0, 0.0, 0.0],  # <- padding
            ]
        )

        grain = create_grain(
            cube_eq,
            create_quad_rule=False,
            create_mesh=False,
        )
        result_eq = array(
            [
                [-1.0, 0.0, 0.0, -1.0],
                [0.0, -1.0, 0.0, -1.0],
                [-0.0, -0.0, -1.0, -1.0],
                [0.0, 0.0, 1.0, -1.0],
                [0.0, 1.0, 0.0, -1.0],
                [1.0, 0.0, 0.0, -1.0],
                [0.0, 0.0, 0.0, 0.0],  # <- is still zero
            ]
        )

        self.assertEqual(grain.volume, 8)
        self.assertIsclose(grain.equations, result_eq)

    def test_003_create_grain_with_quad_rule_and_jit(self):
        cube_eq = array(
            [
                [-1.0,  0, 0.0, -1.0],
                [0.0, -1.0, 0.0, -1.0],
                [-0.0, -0.0, -1.0, -1.0],
                [0.0, 0.0, 1.0, -1.0],
                [0.0, 1.0, 0.0, -1.0],
                [1.0, 0.0, 0.0, -1.0],
            ]
        )
        _create_grain = jit(
            create_grain,
            static_argnames=("create_quad_rule",
                             "elp_degree",
                             "create_mesh",
                             "max_elements",
                             "surface_mesh",)
        )
        grain = _create_grain(
            cube_eq,
            create_quad_rule=True,
            create_mesh=True,
            material_parameters={"A": 0.1}
        )
        result_eq = array(
            [
                [-1.0, 0.0, 0.0, -1.0],
                [0.0, -1.0, 0.0, -1.0],
                [-0.0, -0.0, -1.0, -1.0],
                [0.0, 0.0, 1.0, -1.0],
                [0.0, 1.0, 0.0, -1.0],
                [1.0, 0.0, 0.0, -1.0],
            ]
        )
        self.assertEqual(grain.volume, 8)
        self.assertIsclose(grain.equations, result_eq)
        self.assertEqual(grain.material_parameters["A"], 0.1)
        self.assertIsclose(grain.quad_rule[0].sum(), 8)

    def test_004_sample_grain_with_jit(self):
        key = random.key(0)
        max_elements = 1000
        @jit
        def sample(key):
            return sample_grain(key, 10, create_mesh=True, maxh=0.3, surface_mesh=False,
                                max_elements=max_elements, create_quad_rule=False)
        
        grain = sample(key)
        self.assertEqual(grain.volume, 4.4137897)
        self.assertEqual(grain.mesh.num_elements, 488)
        self.assertEqual(grain.mesh.maxh, 0.3)
        self.assertEqual(grain.mesh.nodes.shape, (max_elements, 3))
        self.assertEqual(grain.mesh.vol_elements.shape, (max_elements, 4))
        self.assertEqual(grain.mesh.sur_elements.shape, (max_elements, 3))
