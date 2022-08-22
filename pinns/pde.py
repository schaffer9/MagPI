from dataclasses import dataclass

from jaxopt import EqualityConstrainedQP
from jaxopt.linear_solve import solve_cg

from .prelude import *


Array = ndarray
Scalar = ndarray
Params = Array
TestFunction = Callable[[Array], Array]
BndFunction = Callable[[Array], Scalar]
SourceFunction = Callable[[Array], Scalar]


@dataclass(frozen=True)
class Poisson:
    sol: TestFunction
    params: Params
    stiffness_matrix: Array
    source_data: Optional[Array] = None
    bnd_data: Optional[Array] = None

    def __call__(self, x: Array) -> Scalar:
        return self.sol(x)


def poisson_dirichlet_ecqp_mc(
    u: TestFunction,
    f: SourceFunction,
    x_dom: Array,
    x_bnd: Array,
    y_bnd: Array,
    stiffness_matrix: Optional[Array]=None,
    init: Optional[Array]=None,
    **solver_kwargs
) -> Poisson:
    r"""Solves the Poisson equation :math:`-\Delta v = f` with :math:`v(x) = p^Tu(x)` and Dirichlet boundary 
    conditions and computes the parameters :math:`p`. The weak formulation is converted to a 
    quadratic program with equality constrains
    
    .. math::
        \min_p \frac{1}{2}p^T S p + b^T p \quad \text{subject to} \; Ap = y,

    where :math:`S` is the stiffness matrix :math:`S_{ij} = int_\Omega\nabla u_i \cdot \nabla u_j dx` and 
    :math:`b_i = \int_\Omega fu_i dx` is the source contribution. The constrains are
    :math:`A_i p = u(x_i) \cdot p = y_{i}` for :math:`i=1, \dots, m` for all :math:`m` boundary values.

    Parameters
    ----------
    u : TestFunction
        A ansatz function :math:`u` such that :math:`v(x) = u(x) \cdot p` can approximate the solution of the PDE.
    f : SourceFunction
    x_dom : Array
    x_bnd : Array
    y_bnd : Array
    stiffness_matrix : Optional[Array], optional
        by default it is computed by Monte Carlo sampling. It is therefore important that
        the data is uniformly distributed over the domain.
    init : Optional[Array], optional
        initial parameters

    Returns
    -------
    Poisson
    """
    data = concatenate([x_bnd, x_dom])
    if stiffness_matrix is None:
        S = compute_stiffness_matrix(u, data)
    else:
        S = stiffness_matrix

    if f is not None:
        source_data = compute_source_data(f, u, data)
    else:
        source_data = None
    
    A = vmap(u)(x_bnd)
    
    eq = EqualityConstrainedQP(**solver_kwargs)
    params = eq.run(
        init_params=init,
        params_obj=(S, source_data),
        params_eq=(A, y_bnd)
    ).params.primal
    
    return Poisson(lambda x: u(x) @ params, params, S, source_data, None)


def poisson_dirichlet_qp_mc(
    u: TestFunction,
    g: BndFunction,
    data: Array,
    f: Optional[SourceFunction]=None, 
    stiffness_matrix: Optional[Array]=None,
    solver: Optional[Callable[..., Array]]=None,
    **solver_kwargs
) -> Poisson:
    r"""Solves the Poisson equation :math:`-\Delta v = f` with :math:`v(x) = g(x) + p^Tu(x)` and Dirichlet boundary 
    conditions and computes the parameters :math:`p`. The weak formulation is converted to a unconstrained
    quadratic program. The test function `u` must be zero at the boundaries and `g` must satisfy the boundary 
    conditions.
    
    Parameters
    ----------
    u : TestFunction
        A ansatz function :math:`u` such that :math:`v(x) = g(x) + u(x) \cdot p` can approximate the solution of the PDE.
        The test function must be zero at the boundaries.
    g : BndFunction
        This function must satisfy the Dirichlet boundary conditions.
    data : Array
        Uniformly distributed sample points from inside the domain.
    f : SourceFunction
    stiffness_matrix : Optional[Array], optional
        by default it is computed by Monte Carlo sampling. It is therefore important that
        the data is uniformly distributed over the domain.
    solver : Optional[Callable[..., Array]]
        
    Returns
    -------
    Poisson
    """
    if stiffness_matrix is None:
        S = compute_stiffness_matrix(u, data)
    else:
        S = stiffness_matrix
    
    bnd_data = compute_bnd_data(u, g, data)
    if f is not None:
        source_data = compute_source_data(f, lambda x: u(x) + g(x), data)
    else:
        source_data = None

    params = poisson_dirichlet_qp(S, bnd_data, source_data, solver, **solver_kwargs)
    return Poisson(lambda x: g(x) + u(x) @ params, params, S, source_data, bnd_data)
    

def compute_bnd_data(u: TestFunction, g: BndFunction, data: Array):
    jac_test_fun = vmap(jacfwd(u))(data)
    grad_g = vmap(grad(g))(data)
    return jnp.tensordot(grad_g, jac_test_fun, ((0,1), (0,2))) / len(data)


def compute_source_data(f: SourceFunction, u: TestFunction, data: Array):
    test_data = vmap(u)(data)
    return -jnp.sum(vmap(f)(data) * jnp.transpose(test_data), axis=1) / len(data)


def compute_stiffness_matrix(u: TestFunction, data: Array):
    Ju = jacfwd(u)
    gradients = vmap(Ju)(data)
    return jnp.tensordot(gradients, gradients, ((0,2), (0,2))) / len(data)


def poisson_dirichlet_qp(
    stiffness_matrix: Array,
    bnd_data: Array,
    source_data: Array=None,
    solver: Callable[..., Array]=None, 
    **solver_kwargs
) -> Params:
    if source_data is None:
        b = bnd_data
    else:
        b = bnd_data + source_data

    if solver is None:
        solver = solve_cg
    
    return solver(lambda p: stiffness_matrix @ p, -b, **solver_kwargs)
    

# todo: add SGD solver