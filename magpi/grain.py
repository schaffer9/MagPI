import dataclasses
from typing import Sequence

import numpy as np
from scipy.spatial import HalfspaceIntersection, ConvexHull
from scipy.optimize import linprog
from jax.tree_util import register_pytree_node_class


from .prelude import *
from .integrate import Weights, Nodes
from .elp import compute_elp, make_elp_quad_rule
from .r_fun import hyperplane_intersection, r0
from .mesh import Mesh, generate_convex_mesh


@register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class Grain:
    vertices: Array
    volume: Array
    equations: Array
    lower_bound: Array
    upper_bound: Array
    mesh: Mesh | None = None
    quad_rule: tuple[Weights, Nodes] | None = None
    material_parameters: dict[str, float] = dataclasses.field(default_factory=dict)

    def tree_flatten(self):
        children = dataclasses.astuple(self)
        return (children, None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


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


_box = np.array(
    [
        [-1.0, 0.0, 0.0, -1.0],
        [0.0, -1.0, 0.0, -1.0],
        [-0.0, -0.0, -1.0, -1.0],
        [0.0, 0.0, 1.0, -1.0],
        [0.0, 1.0, 0.0, -1.0],
        [1.0, 0.0, 0.0, -1.0],
    ]
)


@jit
def _adf(x, equations, r_system=r0, min_val=-1, max_val=1):
    return hyperplane_intersection(equations, r_system=r_system, min_val=min_val, max_val=max_val)(x)


def sample_grain(
    max_faces: int,
    bounds=_box,
    rng: np.random.RandomState | None = None,
    min_offset: float = 0.4,
    max_offset: float = 1.0,
    create_quad_rule: bool = False,
    elp_domain: Array | Sequence[Array] | None = None,
    elp_degree: int = 4,
    quad_rule_kwargs: dict | None = None,
    create_mesh: bool = False,
    meshing_kwargs: dict | None = None,
    material_parameters: dict | None = None,
):
    n = max(max_faces - bounds.shape[0], 0)
    dim = bounds.shape[-1] - 1
    equations = sample_planes(n, dim, rng=rng, min_offset=min_offset, max_offset=max_offset)
    equations = np.concatenate([bounds, equations])
    equations = np.pad(equations, [(0, max_faces - equations.shape[0]), (0, 0)])

    return create_grain(
        equations,
        create_quad_rule=create_quad_rule,
        elp_domain=elp_domain,
        elp_degree=elp_degree,
        quad_rule_kwargs=quad_rule_kwargs,
        create_mesh=create_mesh,
        meshing_kwargs=meshing_kwargs,
        material_parameters=material_parameters,
    )


def create_grain(
    equations: np.ndarray,
    create_quad_rule: bool = False,
    elp_domain: Array | Sequence[Array] | None = None,
    elp_degree: int = 4,
    quad_rule_kwargs: dict | None = None,
    create_mesh: bool = False,
    meshing_kwargs: dict | None = None,
    material_parameters: dict | None = None,
) -> Grain:
    if material_parameters is None:
        material_parameters = {}

    grain = _intersection(equations)
    grain = dataclasses.replace(grain, material_parameters=material_parameters)

    if create_mesh:
        if meshing_kwargs is None:
            meshing_kwargs = {}

        equations = np.asarray(grain.equations)
        mesh = generate_convex_mesh(equations, **meshing_kwargs)
        grain = dataclasses.replace(grain, mesh=mesh)

    if create_quad_rule:
        if quad_rule_kwargs is None:
            quad_rule_kwargs = {}
        if elp_domain is None:
            elp_domain = [array([lb, ub]) for lb, ub in zip(grain.lower_bound, grain.upper_bound)]

        _eq = asarray(equations)
        elp = compute_elp(_adf, elp_domain, elp_degree, _eq, **quad_rule_kwargs)
        quad_rule = make_elp_quad_rule(elp)
        grain = dataclasses.replace(grain, quad_rule=quad_rule)

    grain = shift_grain(grain, -1, 1)
    return grain


def sample_planes(n, dim, rng=None, min_offset=0.4, max_offset=1.0):
    if rng is None:
        rng = np.random

    normal = rng.uniform(-1, 1, (n, dim))
    normal = normal / np.linalg.norm(normal, axis=-1, keepdims=True)
    offset = -rng.uniform(min_offset, max_offset, (n, 1))
    return np.concatenate([normal, offset], axis=-1)


def _intersection(equations):
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
    vol = hull.volume
    return Grain(
        vertices=asarray(vertices),
        volume=asarray(vol),
        equations=asarray(equations),
        lower_bound=asarray(hull.min_bound),
        upper_bound=asarray(hull.max_bound),
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
    n_new = n_new / norm(n_new, axis=-1, keepdims=True)
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
    vertices = _affine(grain.vertices, A, b)
    if grain.mesh is not None:
        nodes = _affine(grain.mesh.nodes, A, b)
        mesh = grain.mesh._replace(nodes=nodes)
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
        vertices=vertices,
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
