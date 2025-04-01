import operator
from dataclasses import dataclass
from typing import Self, NamedTuple

from jax.tree_util import register_pytree_node_class
import numpy as np
from scipy.spatial import HalfspaceIntersection, ConvexHull, QhullError
from scipy.optimize import linprog
try:
    from netgen.csg import CSGeometry, Pnt, Vec, Plane
    from netgen.meshing import MeshingStep
    NGSOLVE_INSTALLED = True
except ImportError:
    NGSOLVE_INSTALLED = False

from .prelude import *
from .integrate import Weights, Nodes


class Mesh(NamedTuple):
    nodes: Array
    elements: Array


class Grain(NamedTuple):
    vertices: Array
    volume: Array
    equations: Array
    surface_mesh: Mesh | None
    quad_rule: tuple[Weights, Nodes]
    lower_bound: Array
    upper_bound: Array


def _eq_to_plane(eq):
    p, v = eq[:-1] * eq[-1], -eq[:-1]
    return Plane(Pnt(*p), Vec(*v))


def generate_mesh(equations, *, maxh=0.25, volume=False):
    if not NGSOLVE_INSTALLED:
        raise ImportError("NGSolve is required for this function but it is not installed.")
    geo = CSGeometry()
    planes = map(_eq_to_plane, equations)
    grain = reduce(operator.mul, planes)
    geo.Add(grain)
    if volume:
        mesh = geo.GenerateMesh(maxh=maxh)
        vertices = np.array([p.p for p in mesh.Points()])
        elements = np.array([[v.nr for v in e.vertices] for e in mesh.Elements3D()]) - 1
    else:
        mesh = geo.GenerateMesh(maxh=maxh, perfstepsend=MeshingStep.MESHSURFACE)
        vertices = np.array([p.p for p in mesh.Points()])
        elements = np.array([[v.nr for v in e.vertices] for e in mesh.Elements2D()]) - 1
    return vertices, elements


def scale_grain(grain: Grain, scaling_factor: float):
    ...


def center_grain(grain: Grain):
    ...


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


def sample_grain(max_faces, bounds=_box, rng=None, min_offset=0.4, max_offset=1.0):
    n = max(max_faces - bounds.shape[0], 0)
    dim = bounds.shape[-1] - 1
    equations = sample_planes(n, dim, rng=rng, min_offset=min_offset, max_offset=max_offset)
    equations = np.concatenate([bounds, equations])
    # TODO: scale between [-1, 1]
    # TODO: add mesh; add quad rule
    return _intersection(equations)


def sample_planes(n, dim, rng=None, min_offset=0.4, max_offset=1.0):
    if rng is None:
        rng = np.random
    
    normal = rng.uniform(-1, 1, (n, dim,))
    normal = normal / np.linalg.norm(normal, axis=-1, keepdims=True)
    offset = -rng.uniform(min_offset, max_offset, (n, 1))
    return np.concatenate([normal, offset], axis=-1)


def _intersection(equations):
    # compute interiour point (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.HalfspaceIntersection.html#scipy.spatial.HalfspaceIntersection)
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
    vertices = hull.points[hull.vertices]
    vol = hull.volume
    return equations, vertices, vol


def _unique_equations(convex_hull):
    eq = np.round(convex_hull.equations, 8)  # round to 8 digits to avoid precision errors
    _, i = np.unique(eq, axis=0, return_index=True)
    eq = convex_hull.equations[i]
    return eq
