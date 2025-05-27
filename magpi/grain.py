import dataclasses
from typing import Sequence
import warnings

import numpy as np
from scipy.spatial import HalfspaceIntersection, ConvexHull
from scipy.optimize import linprog
from jax.tree_util import register_pytree_node_class
from jax.experimental import io_callback

from .prelude import *
from .integrate import Weights, Nodes
from .elp import compute_elp, make_elp_quad_rule
from .r_fun import hyperplane_intersection, r0, ADF, RFun
from .mesh import Mesh, generate_convex_mesh, empty_mesh


Equations = Array


@register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class Grain:
    volume: Array
    equations: Equations
    lower_bound: Array
    upper_bound: Array
    mesh: Mesh | None = None
    quad_rule: tuple[Weights, Nodes] | None = None
    material_parameters: dict[str, float] = dataclasses.field(default_factory=dict)

    def adf(self, x: Array, r_system: RFun = r0, min_val: float = -1, max_val: float = 1):
        return hyperplane_intersection(self.equations, r_system=r_system, min_val=min_val, max_val=max_val)(x)

    def sdf(self, x: Array) -> Array:
        return sdf(x, self.equations)

    def tree_flatten(self):
        children = dataclasses.astuple(self)
        return (children, None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


_box = np.array(  # face equations of a cube [-1, 1]^3
    [
        [-1.0, 0.0, 0.0, -1.0],
        [0.0, -1.0, 0.0, -1.0],
        [-0.0, -0.0, -1.0, -1.0],
        [0.0, 0.0, 1.0, -1.0],
        [0.0, 1.0, 0.0, -1.0],
        [1.0, 0.0, 0.0, -1.0],
    ]
)


def _adf(x: Array, equations, r_system: RFun = r0, min_val: float = -1, max_val: float = 1):
    return hyperplane_intersection(equations, r_system=r_system, min_val=min_val, max_val=max_val)(x)


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
    bounds=_box,
    min_offset: float = 0.4,
    max_offset: float = 1.0,
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
):
    n = max(max_faces - bounds.shape[0], 0)
    dim = bounds.shape[-1] - 1
    equations = sample_planes(key, n, dim, min_offset=min_offset, max_offset=max_offset)
    equations = jnp.concatenate([bounds, equations])

    return create_grain(
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

    grain = intersection_callback(equations)
    grain = dataclasses.replace(grain, material_parameters=material_parameters)

    if create_mesh:
        if meshing_kwargs is None:
            meshing_kwargs = dict(grading=0.5)

        mesh = _generate_mesh_callback(equations, maxh=maxh, max_elements=max_elements, surface_mesh=surface_mesh, **meshing_kwargs)
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

    grain = shift_grain(grain, -1, 1)
    return grain


def _generate_mesh_callback(equations: Array, max_elements: int, surface_mesh: bool, **meshing_kwargs):
    out_mesh = empty_mesh(max_elements, 3, surface_mesh)
    return io_callback(generate_convex_mesh, out_mesh, equations, max_elements=max_elements, 
                       surface_mesh=surface_mesh, **meshing_kwargs)


def intersection_callback(equations: Array):
    out_grain = _empty_grain(equations)
    return io_callback(_intersection, out_grain, equations)


def _empty_grain(equations: Array):
    return Grain(
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


def _intersection(equations: np.ndarray):
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
    equations = np.pad(equations, ((0, n - equations.shape[0]), (0, 0)))  # add padding equations again for equal output size
    vol = hull.volume
    return Grain(
        volume=vol,
        equations=equations,
        lower_bound=hull.min_bound,
        upper_bound=hull.max_bound,
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
    n_new = jnp.where(s > 0, n_new / jnp.where(s > 0, s, 1.0), 0.0)
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
