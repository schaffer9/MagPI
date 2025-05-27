import operator
from typing import NamedTuple, Any, Callable
import warnings

import numpy as np

try:
    from netgen.csg import CSGeometry, Pnt, Vec, Plane
    from netgen.meshing import Mesh as NGMesh, Element2D, MeshPoint, FaceDescriptor
    from ngsolve.webgui import Draw, WebGLScene
    from netgen.meshing import MeshingStep
    NGSOLVE_INSTALLED = True
except ImportError:
    NGSOLVE_INSTALLED = False

from .prelude import *


class Mesh(NamedTuple):
    nodes: Array | np.ndarray
    vol_elements: Array | np.ndarray
    sur_elements: Array | np.ndarray
    num_elements: int
    maxh: float

    def to_jax(self):
        return Mesh(
            nodes=asarray(self.nodes),
            vol_elements=asarray(self.vol_elements),
            sur_elements=asarray(self.sur_elements),
            num_elements=self.num_elements,
            maxh=self.maxh
        )


def generate_convex_mesh(
        equations: Array | np.ndarray, 
        *, 
        maxh: float = 0.2, 
        max_elements: int | None = None,
        surface_mesh: bool = False,
        **kwargs: Any
    ) -> Mesh:
    """Generate a mesh of a convex body with netget

    Parameters
    ----------
    equations : Array | np.ndarray
        normal equations of the convex body
    maxh : float, optional
        global upper bound for mesh size
    max_elements : int | None
        specifies the maximum number of elements allowed,
        if no mesh is found for the given `maxh`, then `maxh` is
        increased until such a mesh can be generated - the mesh
        is then zero padded to this number, defaults to None

    Returns
    -------
    Mesh

    Raises
    ------
    ImportError
        is raised if NGSolve is not installed
    ValueError
        is raised if no mesh could be generated with less than `max_elements`
        within 10 iterations
    """
    if not NGSOLVE_INSTALLED:
        raise ImportError("NGSolve is required for this function but it is not installed.")
    geo = CSGeometry()
    filtered_equations = filter(_padding_equation, equations)
    planes = map(_eq_to_plane, filtered_equations)
    grain = reduce(operator.mul, planes)
    geo.Add(grain)
    for _ in range(10):
        if surface_mesh:
            mesh = geo.GenerateMesh(maxh=maxh, perfstepsend=MeshingStep.MESHSURFACE, **kwargs)
        else:
            mesh = geo.GenerateMesh(maxh=maxh, **kwargs)
        nodes = np.array([p.p for p in mesh.Points()])
        vol_elements = np.array([[v.nr for v in e.vertices] for e in mesh.Elements3D()]) - 1
        sur_elements = np.array([[v.nr for v in e.vertices] for e in mesh.Elements2D()]) - 1
        num_elements = max(vol_elements.shape[0], sur_elements.shape[0])
        if max_elements is None:
            return Mesh(nodes, vol_elements, sur_elements, num_elements, maxh)
        
        if num_elements < max_elements:
            mesh = Mesh(nodes, vol_elements, sur_elements, num_elements, maxh)
            mesh = pad_mesh(mesh, max_elements)
            return mesh
        else:
            maxh *= 2
    
    else:
        raise ValueError(f"Could not generate mesh with less than {max_elements} elements.")


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


def pad_mesh(mesh: Mesh, max_elements: int):
    mesh = Mesh(
        nodes=_pad(mesh.nodes, max_elements),
        vol_elements=_pad(mesh.vol_elements, max_elements),
        sur_elements=_pad(mesh.sur_elements, max_elements),
        num_elements=mesh.num_elements,
        maxh=mesh.maxh
    )
    return mesh


def empty_mesh(max_elements: int, dim: int, surface_mesh: bool) -> Mesh:
    """Returns a empty mesh with all zeros but the right dimensions.
    This is usefull for `io_callback`.

    Parameters
    ----------
    max_elements : int
    dim : int
    surface_mesh : bool

    Returns
    -------
    Mesh
    """
    return Mesh(
        nodes=zeros((max_elements, dim)),
        vol_elements=array([]) if surface_mesh else zeros((max_elements, dim + 1), dtype=jnp.int32),
        sur_elements=zeros((max_elements, dim), dtype=jnp.int32),
        num_elements=asarray(max_elements),
        maxh=asarray(0.0),
    )
    


def _pad(a: np.ndarray, max_length: int):
    if a.shape == () or a.shape == (0,):
        return a
    return np.pad(a, ((0, max_length - a.shape[0]), *[(0, 0)] * (a.ndim - 1)))
