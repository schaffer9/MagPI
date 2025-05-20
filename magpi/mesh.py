import operator
from typing import NamedTuple, Any

import numpy as np

try:
    from netgen.csg import CSGeometry, Pnt, Vec, Plane
    from netgen.meshing import Mesh as NGMesh, Element2D, MeshPoint, FaceDescriptor
    from ngsolve.webgui import Draw, WebGLScene
    NGSOLVE_INSTALLED = True
except ImportError:
    NGSOLVE_INSTALLED = False

from .prelude import *


class Mesh(NamedTuple):
    nodes: Array
    vol_elements: Array
    sur_elements: Array


def generate_convex_mesh(equations: Array | np.ndarray, *, maxh: float = 0.25, **kwargs: Any) -> Mesh:
    """Generate a mesh of a convex body with netget

    Parameters
    ----------
    equations : Array | np.ndarray
        normal equations of the convex body
    maxh : float, optional
        global upper bound for mesh size.

    Returns
    -------
    Mesh

    Raises
    ------
    ImportError
        is raised if NGSolve is not installed
    """
    if not NGSOLVE_INSTALLED:
        raise ImportError("NGSolve is required for this function but it is not installed.")
    geo = CSGeometry()
    filtered_equations = filter(_padding_equation, equations)
    planes = map(_eq_to_plane, filtered_equations)
    grain = reduce(operator.mul, planes)
    geo.Add(grain)
    mesh = geo.GenerateMesh(maxh=maxh, **kwargs)
    nodes = np.array([p.p for p in mesh.Points()])
    vol_elements = np.array([[v.nr for v in e.vertices] for e in mesh.Elements3D()]) - 1
    sur_elements = np.array([[v.nr for v in e.vertices] for e in mesh.Elements2D()]) - 1
    return Mesh(asarray(nodes), asarray(vol_elements), asarray(sur_elements))


def draw_mesh(mesh: Mesh, *args: Any, show: bool = True, **kwargs: Any) -> WebGLScene | None:
    """Draws a mesh using WebGLScene

    Parameters
    ----------
    mesh : Mesh
    show : bool, optional
        by default True

    Returns
    -------
    WebGLScene | None
    
    Raises
    ------
    ImportError
        is raised if NGSolve is not installed
    """
    if not NGSOLVE_INSTALLED:
        raise ImportError("NGSolve is required for this function but it is not installed.")
    # regenerate surface mesh
    ngmesh = NGMesh(3)
    ngmesh.Add(FaceDescriptor(surfnr=1, domin=1, bc=1))
    pnums = []
    for pnt in mesh.nodes:
        pnums.append(ngmesh.Add(MeshPoint(Pnt(*pnt))))

    for e2d in mesh.sur_elements:
        ngmesh.Add(Element2D(1, [pnums[i] for i in e2d]))

    return Draw(ngmesh, *args, show=show, **kwargs)


def _padding_equation(eq):
    n = eq[:-1]
    return ~np.isclose(np.linalg.norm(n), 0.0)


def _eq_to_plane(eq):
    p, v = eq[:-1] * eq[-1], -eq[:-1]
    return Plane(Pnt(*p), Vec(*v))
