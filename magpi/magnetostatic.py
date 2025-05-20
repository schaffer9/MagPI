import dataclasses
from typing import Callable, Any, Protocol

from jax.tree_util import register_pytree_node_class

from .prelude import *
from .calc import laplace, divergence, curl, value_and_jacfwd
from .slp import (
    charge_tensor_for_mesh,
    source_tensor_for_mesh,
    single_layer_potential,
    curl_single_layer_potential,
    vector_potential_charge,
    scalar_potential_charge,
)
from .mesh import Mesh
from .r_fun import ADF

Weights = Array
Nodes = Array
QuadRule = tuple[Weights, Nodes]
Mag = Callable[..., Array]
Residual = Array
Scalar = Array | float

ELM = Callable[[Array], Array]
ElmParams = Array


@register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class ElmPoissonSolution:
    elm_params: Array
    strong_residual: Array
    
    def tree_flatten(self):
        children = (self.elm_params, self.strong_residual)  # arrays / dynamic values
        return (children, None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class ElmPoissonSolver:
    _adf: ADF
    adf_args: tuple[Any]
    elm: ELM
    quad_rule: QuadRule
    Q: Array
    Pinv: Array
    condition_number: Array

    def solve(self, f: Callable[..., Array], *args, **kwargs) -> ElmPoissonSolution:
        W, X = self.quad_rule[0].reshape(-1), self.quad_rule[1].reshape(-1, 3)
        _f = lambda x: f(x, *args, **kwargs)
        b = vmap(_f)(X)
        _b = sqrt(W)[:, *[None for _ in b.shape[1:]]] * b
        elm_params = self.Pinv @ _b
        lap_phi1 = -(self.Q @ elm_params)
        residuals = (lap_phi1 + b)
        if residuals.ndim > 1:
            residuals = norm(residuals, axis=tuple(range(1, residuals.ndim)))
        strong_residual = jnp.sqrt(jnp.sum(W * residuals ** 2))
        return ElmPoissonSolution(elm_params, strong_residual)

    def adf(self, x: Array) -> Scalar:
        return self._adf(x, *self.adf_args)
    
    def u(self, x: Array, elm_params: ElmParams) -> Array:
        return self.adf(x) * self.elm(x) @ elm_params
    
    def tree_flatten(self):
        children = (self.adf_args, self.quad_rule, self.Q, self.Pinv, self.condition_number)
        aux_data = (self._adf, self.elm)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(aux_data[0], children[0], aux_data[1], children[1], children[2], children[3], children[4])


def create_elm_poisson_solver(
    adf: ADF,
    elm: ELM,
    quad_rule: QuadRule,
    *adf_args: Any,
    eps: float = 1e-4
) -> ElmPoissonSolver:
    r"""Creates a solver for a Poisson problem
    with homogeneous boundary conditions
    
    .. math::
        -\Delta u = f \;\; \text{for}\;x \in \Omega \\
        u = 0 \;\; \text{for}\;x \in \partial\Omega
        
    using hard constraint ELMs ansatz :math:`u_{\boldsymbol\beta}(x)=\mathrm{adf}(x)\mathrm{elm}(x)^T\boldsymbol\beta`.

    Parameters
    ----------
    adf : ADF
        Approximate Distance Function
    elm : ELM
        Extreme Learning Machine embedding for the solver
    quad_rule : QuadRule
        quadrature rule to integrate over the domain
    eps : float, optional
        cut off tolerance for singular values, by default 1e-4

    Returns
    -------
    ElmPoissonSolver
    """
    W, X = quad_rule[0].reshape(-1), quad_rule[1].reshape(-1, 3)
    u_phi1 = lambda x: elm(x) * adf(x, *adf_args)
    Q = lax.map(lambda x: -laplace(u_phi1)(x), X, batch_size=1000)
    U, S, VT = jax.scipy.linalg.svd(sqrt(W[:, None]) * Q, full_matrices=False, lapack_driver="gesvd")
    Sinv = jnp.where(S > eps, 1 / jnp.where(S > eps, S, jnp.inf), 0.0)
    condition_number = S[0] / jnp.maximum(S[-1], eps)
    Pinv = VT.T * Sinv @ U.T
    return ElmPoissonSolver(_adf=adf, adf_args=adf_args, elm=elm, quad_rule=quad_rule, 
                            Q=Q, Pinv=Pinv, condition_number=condition_number)


@register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class SlpSolver:
    mesh: Mesh
    tri_quad_rule: QuadRule
    order: int = 2

    def compute_source(self, x, compute_jacfwd: bool = True):
        return source_tensor_for_mesh(
            x, self.mesh, self.tri_quad_rule[0], self.tri_quad_rule[1], compute_jacfwd=compute_jacfwd, order=self.order
        )

    def compute_charge(self, charge_fn: Callable, *args, **kwargs) -> Array:
        f = lambda x, n: charge_fn(x, n, *args, **kwargs)
        return charge_tensor_for_mesh(f, self.mesh, order=self.order)

    def scalar_potential_charge(self, mag: Callable, phi1_elm: Callable, *args, normalized: bool = True, **kwargs):
        charge_fn = scalar_potential_charge(mag, phi1_elm, normalized=normalized)
        return self.compute_charge(charge_fn, *args, **kwargs)

    def vector_potential_charge(self, mag: Callable, A1_elm: Callable, *args, normalized: bool = True, **kwargs):
        charge_fn = vector_potential_charge(mag, A1_elm, normalized=normalized)
        return self.compute_charge(charge_fn, *args, **kwargs)

    def slp(self, source: Array, charges: Array) -> Array:
        return single_layer_potential(source, charges)

    def grad_slp(self, d_source: Array, charges: Array) -> Array:
        _grad_slp = single_layer_potential(d_source, charges)
        assert _grad_slp.shape == (3,)
        return _grad_slp

    def curl_slp(self, d_source: Array, charges: Array) -> Array:
        _curl_slp = curl_single_layer_potential(d_source, charges)
        assert _curl_slp.shape == (3,)
        return _curl_slp
    
    def tree_flatten(self):
        children = (self.mesh, self.tri_quad_rule)  # arrays / dynamic values
        aux_data = self.order
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children[0], children[1], aux_data)


def create_slp_solver(mesh: Mesh, tri_quad_rule: QuadRule, order: int = 2):
    return SlpSolver(mesh=mesh, tri_quad_rule=tri_quad_rule, order=order)


@register_pytree_node_class
@dataclasses.dataclass
class Potential:
    X: Array  # evaluation points
    Z: Array  # sources at evaluation points
    dZ: Array  # gradient of sources
    u1: Array  # first part of potential at X given by homogeneous poisson
    u2: Array  # second part of potential at x given by slp
    field1: Array  # first part of magnetostatic field at X
    field2: Array  # second part of magnetostatic field at X
    u1_solution: ElmPoissonSolution  # ELM solution for first part of potential

    @property
    def field(self) -> Array:
        return self.field1 + self.field2

    @property
    def potential(self) -> Array:
        return self.u1 + self.u2

    def tree_flatten(self):
        children = (self.X, self.Z, self.dZ, self.u1, self.u2, self.field1, self.field2, self.u1_solution)
        return (children, None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
    
    
class PotentialSolver(Protocol):
    def solve(
        self,
        mag: Mag,
        X: Array | tuple[Array, Array, Array],
        *args: Any,
        u1_solution: ElmPoissonSolution | None = None,
        **kwargs: Any,
    ) -> Potential:
        """Full solution of the potential together with magnetostatic fields.

        Parameters
        ----------
        mag : Mag
        X : Array | tuple[Array, Array, Array]
        u1_solution : ElmPoissonSolution | None, optional
            this can be used to avoid recomputation of u1, by default None

        Returns
        -------
        Potential
        """
        ...
        
    def u1_solution(self, mag: Mag, *args: Any, **kwargs: Any) -> ElmPoissonSolution:
        """Solution of the first part of the potential.

        Parameters
        ----------
        mag : Mag

        Returns
        -------
        ElmPoissonSolution
        """
        ...


@register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class ScalarPotentialSolver:
    poisson_solver: ElmPoissonSolver
    slp_solver: SlpSolver

    def solve(
        self,
        mag: Mag,
        sources: Array | tuple[Array, Array, Array],
        *args: Any,
        u1_solution: ElmPoissonSolution | None = None,
        **kwargs: Any,
    ) -> Potential:
        _mag = lambda x: mag(x, *args, **kwargs)

        if isinstance(sources, tuple):
            X, Z, dZ = sources[0], sources[1], sources[2]
        else:
            X = sources
            Z, dZ = vmap(self.slp_solver.compute_source)(sources)

        if u1_solution is None:
            u1_solution = self.u1_solution(_mag)
        phi1 = lambda x: self.poisson_solver.u(x, u1_solution.elm_params)
        phi1_x, h1_x = vmap(value_and_jacfwd(phi1))(X)
        charges = self.slp_solver.scalar_potential_charge(_mag, phi1, normalized=False)
        phi2_x = vmap(lambda z: self.slp_solver.slp(z, charges))(Z)
        h2_x = vmap(lambda dz: self.slp_solver.grad_slp(dz, charges))(dZ)
        return Potential(X, Z, dZ, phi1_x, phi2_x, -h1_x, -h2_x, u1_solution)

    def u1_solution(self, mag: Mag, *args: Any, **kwargs: Any) -> ElmPoissonSolution:
        _mag = lambda x: mag(x, *args, **kwargs)
        u1_solution = self.poisson_solver.solve(lambda x: -asarray(divergence(_mag)(x)))
        return u1_solution

    def tree_flatten(self):
        children = (self.poisson_solver, self.slp_solver)
        return (children, None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class VectorPotentialSolver:
    poisson_solver: ElmPoissonSolver
    slp_solver: SlpSolver

    def solve(
        self,
        mag: Mag,
        sources: Array | tuple[Array, Array, Array],
        *args: Any,
        u1_solution: ElmPoissonSolution | None = None,
        **kwargs: Any,
    ):
        _mag = lambda x: mag(x, *args, **kwargs)

        if isinstance(sources, tuple):
            X, Z, dZ = sources[0], sources[1], sources[2]
        else:
            X = sources
            Z, dZ = vmap(self.slp_solver.compute_source)(sources)

        if u1_solution is None:
            u1_solution = self.u1_solution(_mag)

        A1 = lambda x: self.poisson_solver.u(x, u1_solution.elm_params)
        A1_x, b1_x = vmap(A1)(X), asarray(vmap(curl(A1))(X))
        charges = self.slp_solver.vector_potential_charge(mag, A1, normalized=False)
        A2_x = vmap(lambda z: self.slp_solver.slp(z, charges))(Z)
        b2_x = vmap(lambda dz: self.slp_solver.curl_slp(dz, charges))(dZ)
        return Potential(X, Z, dZ, A1_x, A2_x, b1_x, b2_x, u1_solution)

    def u1_solution(self, mag: Mag, *args: Any, **kwargs: Any) -> ElmPoissonSolution:
        _mag = lambda x: mag(x, *args, **kwargs)
        u1_solution = self.poisson_solver.solve(lambda x: asarray(curl(_mag)(x)))
        return u1_solution
    
    def tree_flatten(self):
        children = (self.poisson_solver, self.slp_solver)
        return (children, None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


def create_scalar_potential_solver(
    adf: Callable,
    elm: Callable,
    quad_rule: QuadRule,
    mesh: Mesh,
    tri_quad_rule: QuadRule,
    *adf_args: Any,
    eps: float = 1e-4,
    order: int = 2,
) -> ScalarPotentialSolver:
    phi1_solver = create_elm_poisson_solver(adf, elm, quad_rule, *adf_args, eps=eps)
    slp_solver = create_slp_solver(mesh, tri_quad_rule, order=order)
    return ScalarPotentialSolver(phi1_solver, slp_solver)


def create_vector_potential_solver(
    adf: Callable,
    elm: Callable,
    quad_rule: QuadRule,
    mesh: Mesh,
    tri_quad_rule: QuadRule,
    *adf_args: Any,
    eps: float = 1e-4,
    order: int = 2,
) -> VectorPotentialSolver:
    phi1_solver = create_elm_poisson_solver(adf, elm, quad_rule, *adf_args, eps=eps)
    slp_solver = create_slp_solver(mesh, tri_quad_rule, order=order)
    return VectorPotentialSolver(phi1_solver, slp_solver)


def to_skew_simmetric_matrix(x):
    S = zeros((3, 3))
    S = S.at[1, 0].set(x[2])
    S = S.at[2, 0].set(-x[1])
    S = S.at[2, 1].set(x[0])
    S = S - S.T
    return S


def cayley_rotation(p, x):
    assert p.shape[0] == 3, f"{p.shape}"
    Q = to_skew_simmetric_matrix(p)
    I = jnp.eye(3)
    return jnp.linalg.inv(I - Q) @ (I + Q) @ x


def elm_mag_model(elm):
    def mag(x, params, m0):
        if callable(m0):
            _m0 = m0(x)
        else:
            _m0 = m0
        p = elm(x) @ params
        return cayley_rotation(p, _m0)

    return mag
