import dataclasses
from typing import Sequence, NamedTuple, Callable, Any

import numpy as np
from scipy.spatial import HalfspaceIntersection, ConvexHull
from scipy.optimize import linprog
from jax.tree_util import register_dataclass
from jax.experimental import io_callback
from jax.scipy import stats

from .prelude import *
from .elp import compute_elp, make_elp_quad_rule
from .r_fun import hyperplane_intersection, r0, ADF, RFun
from .mesh import Mesh, generate_convex_mesh, empty_mesh
from .magnetostatic import (
    ELM,
    PotentialSolver,
    create_scalar_potential_solver,
    create_vector_potential_solver,
    ElmPoissonSolution,
    Potential,
    QuadRule,
    cayley_transform
)
from .sampling import (
    sample_domain,
    sample_magnetization_states,
    Mag,
    default_mag_model,
    draw_mag_params,
    MagParams,
    PDF,
    rejection_sampling
)


Equations = Array


@partial(register_dataclass, 
         data_fields=["volume", "equations", "lower_bound", "upper_bound", 
                      "mesh", "quad_rule", "material_parameters"],
         meta_fields=[])
@dataclasses.dataclass
class Grain:
    volume: Array
    equations: Equations
    lower_bound: Array
    upper_bound: Array
    mesh: Mesh | None = None
    quad_rule: QuadRule | None = None
    material_parameters: dict[str, float] = dataclasses.field(default_factory=dict)

    def adf(self, x: Array, r_system: RFun = r0, min_val: float = -1, max_val: float = 1):
        return grain_adf(x, self.equations, r_system=r_system, min_val=min_val, max_val=max_val)

    def sdf(self, x: Array) -> Array:
        return sdf(x, self.equations)


@partial(register_dataclass, data_fields=["grain", "solver"], meta_fields=[])
@dataclasses.dataclass
class GrainSolver:
    grain: Grain
    solver: PotentialSolver

    def solve(
        self,
        mag: Mag,
        X: Array | tuple[Array, Array, Array],
        *args: Any,
        u1_solution: ElmPoissonSolution | None = None,
        **kwargs: Any,
    ) -> Potential:
        return self.solver.solve(mag, X, *args, u1_solution=u1_solution, **kwargs)


def create_grain_solver(
    grain: Grain,
    elm: ELM,
    quad_rule: QuadRule,
    tri_quad_rule: QuadRule,
    eps: float = 1e-4,
    order: int = 2,
    potential: str = "scalar",
) -> GrainSolver:
    if potential == "scalar":
        assert grain.mesh is not None, "You need to provide a surface mesh to create a solver"
        solver = create_scalar_potential_solver(
            grain_adf, elm, quad_rule, grain.mesh, tri_quad_rule, grain.equations, eps=eps, order=order
        )
    elif potential == "vector":
        assert grain.mesh is not None, "You need to provide a surface mesh to create a solver"
        solver = create_vector_potential_solver(
            grain_adf, elm, quad_rule, grain.mesh, tri_quad_rule, grain.equations, eps=eps, order=order
        )
    else:
        raise ValueError("`potential` must either be 'scalar' or 'vector.")
    return GrainSolver(grain, solver)


def make_mc_quad_rule_for_grain(
    key: Array, collocation_points: int, grain: Grain, curvature_threshold: float = 100
) -> QuadRule:
    X_solver = sample_domain(
        key,
        collocation_points,
        grain.adf,
        lower_bound=grain.lower_bound,
        upper_bound=grain.upper_bound,
        curvature_threshold=curvature_threshold,
    )
    W_solver = ones((X_solver.shape[0],)) / X_solver.shape[0] * grain.volume
    return (W_solver, X_solver)


def sample_grain_domain(key: Array, n_samples: int, grain: Grain, eps: float = 1e-2, curvature_threshold: float = 100):
    return sample_domain(
        key,
        n_samples,
        grain.adf,
        lower_bound=grain.lower_bound,
        upper_bound=grain.upper_bound,
        eps=eps,
        curvature_threshold=curvature_threshold,
    )


def default_pdf(x, mean=zeros((3,)), cov=jnp.identity(3) * 5):
    return stats.multivariate_normal.pdf(x, mean, cov)


def sample_grain_exterior(
    key: Array,
    n_samples: int,
    grain: Grain,
    lower_bound: Array = asarray(-20),
    upper_bound: Array = asarray(20),
    eps: float = 1e-2,
    curvature_threshold: float = 100,
    pdf: PDF = default_pdf,
) -> Array:
    return sample_domain(
        key,
        n_samples,
        grain.adf,
        dimension=2,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        eps=eps,
        curvature_threshold=curvature_threshold,
        pdf=pdf,
    )


def sample_mag_for_grain(
    key: Array,
    n: int,
    grain_solver: GrainSolver,
    mag_model: Mag = default_mag_model,
    mag_params_sample_fn: Callable = draw_mag_params,
    max_exchange_energy: float = 15,
    tol: float = 1e-2,
) -> tuple[MagParams, ElmPoissonSolution, Array]:
    return sample_magnetization_states(
        key, n, grain_solver.solver, mag_model, mag_params_sample_fn, max_exchange_energy, tol
    )


unit_cube = array(  # face equations of a cube [-1, 1]^3
    [
        [-1.0, 0.0, 0.0, -0.5],
        [0.0, -1.0, 0.0, -0.5],
        [-0.0, -0.0, -1.0, -0.5],
        [0.0, 0.0, 1.0, -0.5],
        [0.0, 1.0, 0.0, -0.5],
        [1.0, 0.0, 0.0, -0.5],
    ]
)


def sdf(x: Array, equations: Equations) -> Array:
    """The signed distance function for a convex domain given.
    This is based on the R1-system with min and max being conjunction
    and disjunction.

    Note
    ----
    This signed distance function associative and can be evaluated way more
    efficiently than other R-functions. This is particularly important for the
    computation of an Equivalent Legendre Polynomial.

    Parameters
    ----------
    x : Array
    equations : Equations
    """
    n, b = equations[:, :3], equations[:, 3]
    r = norm(equations)  # for padding
    d = (n @ x) - b
    d = jnp.where(r == 0, 1, d)  # set padding eqations to 1
    return jnp.min(d)


def sample_grain(
    key: Array,
    max_faces: int,
    bounds: Array = unit_cube,
    min_offset: float = 0.2,
    max_offset: float = 0.8660254,
    keep_aspect_ratio: bool = True,
    create_quad_rule: bool = False,
    elp_domain: Array | Sequence[Array] | None = None,
    elp_degree: int = 4,
    quad_rule_kwargs: dict | None = None,
    adf: ADF = sdf,
    create_mesh: bool = False,
    maxh: float = 0.2,
    max_elements: int = 500,
    surface_mesh: bool = True,
    meshing_kwargs: dict | None = None,
    material_parameters: dict | None = None,
) -> Grain:
    def _sample_grain(key):
        k1, k2, k3, k4 = random.split(key, 4)
        n = max(max_faces - bounds.shape[0], 0)
        i = random.randint(k1, (), minval=0, maxval=n)
        mask = random.choice(k2, array([True, False]), (n,), p=array([i / n, 1 - i / n]))
        mask = jnp.sort(mask, descending=True)
        dim = bounds.shape[-1] - 1
        a, b = jnp.sort(random.uniform(k3, (2,), minval=min_offset, maxval=max_offset))  # offsets
        equations = sample_planes(k4, n, dim, min_offset=a, max_offset=b)
        equations = jnp.where(mask[:, None], equations, 0.0)
        equations = jnp.concatenate([bounds, equations])
        
        # for rotation:
        #p = random.uniform(k5, (3,), minval=-2 * pi, maxval=2 * pi)
        #rot_matrix = cayley_transform(p)
        #equations = _affine_transformation_equations(equations, rot_matrix, zeros((3,)))
        # grain = create_grain(equations)
        # grain = center_grain(grain, keep_aspect_ratio=keep_aspect_ratio)
        # equations = grain.equations
        
        grain = create_grain(
            equations,
            create_quad_rule=create_quad_rule,
            elp_domain=elp_domain,
            elp_degree=elp_degree,
            quad_rule_kwargs=quad_rule_kwargs,
            adf=adf,
            create_mesh=create_mesh,
            maxh=maxh,
            max_elements=max_elements,
            surface_mesh=surface_mesh,
            meshing_kwargs=meshing_kwargs,
            material_parameters=material_parameters,
        )
        return grain
    
    def _accept_grain(grain, key):
        if grain.mesh is not None:
            return grain.mesh.maxh <= (maxh + 1e-2)
        else:
            return asarray(True)
        
    grain = rejection_sampling(key, 1, _sample_grain, _accept_grain)
    return grain


def create_grain(
    equations: Array,
    create_quad_rule: bool = False,
    elp_domain: Array | Sequence[Array] | None = None,
    elp_degree: int = 4,
    quad_rule_kwargs: dict | None = None,
    adf: ADF = sdf,
    create_mesh: bool = False,
    maxh: float = 0.2,
    max_elements: int = 500,
    surface_mesh: bool = True,
    meshing_kwargs: dict | None = None,
    material_parameters: dict | None = None,
) -> Grain:
    if material_parameters is None:
        material_parameters = {}

    grain = Grain(**intersection_callback(equations))
    grain = dataclasses.replace(grain, material_parameters=material_parameters)

    if create_mesh:
        if meshing_kwargs is None:
            meshing_kwargs = dict(grading=0.5)

        mesh = _generate_mesh_callback(
            equations, maxh=maxh, max_elements=max_elements, surface_mesh=surface_mesh, **meshing_kwargs
        )
        grain = dataclasses.replace(grain, mesh=mesh)

    if create_quad_rule:
        if quad_rule_kwargs is None:
            quad_rule_kwargs = {}
        if elp_domain is None:
            elp_domain = [array([lb, ub]) for lb, ub in zip(grain.lower_bound, grain.upper_bound)]

        _eq = asarray(equations)
        elp = compute_elp(adf, elp_domain, elp_degree, _eq, **quad_rule_kwargs)
        quad_rule = make_elp_quad_rule(elp)
        grain = dataclasses.replace(grain, quad_rule=quad_rule)

    return grain


def _generate_mesh_callback(equations: Array, max_elements: int, surface_mesh: bool, **meshing_kwargs):
    out_mesh = empty_mesh(max_elements, 3, surface_mesh)
    return io_callback(
        generate_convex_mesh,
        out_mesh,
        equations,
        max_elements=max_elements,
        surface_mesh=surface_mesh,
        **meshing_kwargs,
    )


def intersection_callback(equations: Array):
    out_grain = _empty_grain(equations)
    return io_callback(_intersection, out_grain, equations)


def _empty_grain(equations: Array) -> dict[str, Array]:
    return dict(
        volume=asarray(0.0),
        equations=zeros_like(equations),
        lower_bound=zeros((3,)),
        upper_bound=zeros((3,)),
    )


def sample_planes(key: Array, n: int, dim: int, min_offset: float = 0.4, max_offset: float = 1.0) -> Equations:
    k1, k2 = random.split(key)
    normal = random.uniform(k1, (n, dim), minval=-1, maxval=1)
    normal = normal / norm(normal, axis=-1, keepdims=True)
    offset = -random.uniform(k2, (n, 1), minval=min_offset, maxval=max_offset)
    return jnp.concatenate([normal, offset], axis=-1)


def _intersection(equations: np.ndarray) -> dict[str, np.ndarray]:
    n = equations.shape[0]
    equations = equations[~np.all(equations == 0, axis=1)]  # remove padding equations
    # compute interiour point
    # (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.HalfspaceIntersection.html#scipy.spatial.HalfspaceIntersection)
    norm_vector = np.reshape(np.linalg.norm(equations[:, :-1], axis=1), (equations.shape[0], 1))
    c = np.zeros((equations.shape[1],))
    c[-1] = -1
    A = np.hstack((equations[:, :-1], norm_vector))
    b = -equations[:, -1:]
    res = linprog(c, A_ub=A, b_ub=b, bounds=(None, None))
    p = res.x[:-1]

    intersection = HalfspaceIntersection(equations, p)
    vertices = intersection.intersections
    hull = ConvexHull(vertices, qhull_options="Q5")
    equations = _unique_equations(hull)
    equations = np.pad(
        equations, ((0, n - equations.shape[0]), (0, 0))
    )  # add padding equations again for equal output size
    vol = hull.volume
    return dict(
        volume=np.asarray(vol),
        equations=np.asarray(equations),
        lower_bound=np.asarray(hull.min_bound),
        upper_bound=np.asarray(hull.max_bound),
    )


def _unique_equations(convex_hull):
    eq = np.round(convex_hull.equations, 8)  # round to 8 digits to avoid precision errors
    _, i = np.unique(eq, axis=0, return_index=True)
    eq = convex_hull.equations[i]
    return eq


def _affine(x, A, b):
    return x @ A.T + b


def _affine_transformation_equations(eq, A, b):
    n, offset = eq[:, :-1], eq[:, -1]
    Ainv = jnp.linalg.inv(A)
    n_new = n @ Ainv
    s = norm(n_new, axis=-1, keepdims=True)
    n_new = asarray(jnp.where(s > 0, n_new / jnp.where(s > 0, s, 1.0), 0.0))
    p = n * offset[:, None]
    p_new = _affine(p, A, -b)
    offset_new = jnp.sum(p_new * n_new, axis=-1)
    return jnp.concatenate([n_new, offset_new[:, None]], axis=-1)


def affine_transformation_on_grain(grain: Grain, A: Array, b: Array) -> Grain:
    det_A = jnp.linalg.det(A)
    eq = _affine_transformation_equations(grain.equations, A, b)
    vol = grain.volume * det_A
    lb = _affine(grain.lower_bound, A, b)
    ub = _affine(grain.upper_bound, A, b)
    if grain.mesh is not None:
        nodes = _affine(grain.mesh.nodes, A, b)
        mesh = dataclasses.replace(grain.mesh, nodes=nodes)
    else:
        mesh = None

    if grain.quad_rule is not None:
        weights, nodes = grain.quad_rule
        nodes = _affine(nodes, A, b)
        weights = weights * det_A
        quad_rule = (weights, nodes)
    else:
        quad_rule = None

    return dataclasses.replace(
        grain,
        volume=vol,
        equations=eq,
        lower_bound=lb,
        upper_bound=ub,
        mesh=mesh,
        quad_rule=quad_rule,
    )


def shift_grain(grain: Grain, lb_new: float | Array, ub_new: float | Array) -> Grain:
    lb, ub = grain.lower_bound, grain.upper_bound
    k = (ub_new - lb_new) / (ub - lb)
    A = diag(k)
    b = lb_new - k * lb
    return affine_transformation_on_grain(grain, A, b)


def center_grain(grain: Grain, keep_aspect_ratio: bool = True) -> Grain:
    lb, ub = grain.lower_bound, grain.upper_bound
    if keep_aspect_ratio:
        d = jnp.max(ub - lb)
    else:
        d = ub - lb
    scaling_factor = 1 / d
    center = (ub + lb) / 2
    centerd_lb = lb - center
    centered_ub = ub - center
    lb_new = centerd_lb * scaling_factor
    ub_new = centered_ub * scaling_factor
    return shift_grain(grain, lb_new, ub_new)


def scale_grain(grain: Grain, scaling_factor: float | Array) -> Grain:
    lb, ub = grain.lower_bound, grain.upper_bound
    lb_new, ub_new = scaling_factor * lb, scaling_factor * ub
    return shift_grain(grain, lb_new, ub_new)


def polytope_intersection(vertices1, vertices2):
    hull1 = vertices1
    if isinstance(vertices1, ConvexHull):
        hull1 = vertices1
        vertices1 = hull1.points[hull1.vertices]
    else:
        hull1 = ConvexHull(vertices1)
    if isinstance(vertices2, ConvexHull):
        hull2 = vertices2
        vertices2 = hull2.points[hull2.vertices]
    else:
        hull2 = ConvexHull(vertices2)

    eq1 = _unique_equations(hull1)
    eq2 = _unique_equations(hull2)
    equations = np.concatenate([eq1, eq2], axis=0)
    return _intersection(equations)


def grain_adf(x: Array, equations: Array, r_system: RFun = r0, min_val: float = -1, max_val: float = 1):
    return hyperplane_intersection(equations, r_system=r_system, min_val=min_val, max_val=max_val)(x)
