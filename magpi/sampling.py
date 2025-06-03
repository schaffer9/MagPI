from typing import Callable, TypeVar, Any
import dataclasses

from scipy.stats.qmc import PoissonDisk
from jax.tree_util import register_dataclass

from .prelude import *
from .quaternions import from_euler_angles, quaternion_rotation
from .magnetostatic import cayley_rotation, PotentialSolver, ElmPoissonSolution, exchange_energy
from .calc import laplace
from .r_fun import ADF


Sample = TypeVar("Sample", covariant=True)
Key = Array
Accept_fn = Callable[[Sample, Key], Array]
SampleFn = Callable[[Key], Sample]
ELM = Callable[..., Array]

Errors = Array
Scalar = float | Array
PDF = Callable[..., Scalar]


def rejection_sampling(
    key: Key,
    n: int,
    sample_fn: SampleFn[Sample],
    accept_fn: Accept_fn[Sample],
) -> Sample:
    """Draws `n` samples according to the given sample function `sample_fn` and accepts the sample
    if `accept_fn` yields True.

    Parameters
    ----------
    key : Key
    n : int
    sample_fn : SampleFn
    accept_fn : Accept_fn
    """

    def draw_sample(key):
        k1, k2, samplekey = random.split(key, 3)
        sample = sample_fn(samplekey)

        def body(state):
            (k1, k2), sample = state
            k1, k2, samplekey = random.split(k1, 3)
            sample = sample_fn(samplekey)
            return (k1, k2), sample

        def not_valid(state):
            (_, k2), sample = state
            return jnp.logical_not(accept_fn(sample, k2))

        _, sample = lax.while_loop(not_valid, body, ((k1, k2), sample))
        return sample

    if n == 1:
        return draw_sample(key)
    else:
        keys = random.split(key, n)
        return vmap(draw_sample)(keys)


def rejection_sampling_from_pdf(key, n: int, pdf: PDF, sample_fn: SampleFn, m: int = 2):
    """Draws `n` samples according to the given probability density function `pdf`.

    Parameters
    ----------
    key : Key
    n : int
    pdf : PDF
    sample_fn : SampleFn
    m : int
    """

    def accept_fn(sample, key):
        p = random.uniform(key)
        return p < (pdf(sample) / m)

    return rejection_sampling(key, n, sample_fn, accept_fn)


uniform_state = lambda x: zeros_like(x).at[..., -1].set(0)

unit_vec = lambda x: x / norm(x, keepdims=True)


def flower_state(x, a: Scalar = 1.0, b: Scalar = 2.0, c: Scalar = 1.0) -> Array:
    mx = 1 / a * x[..., 0] * x[..., 2]
    my = 1 / c * x[..., 1] * x[..., 2] + (1 / b**3 * x[..., 1] * x[..., 2]) ** 3
    mz = ones_like(my)
    mag = stack([mx, my, mz], axis=-1)
    return unit_vec(mag)


def vortex_state(x, rc: Scalar = 0.14) -> Array:
    x, _, z = x[..., 0], x[..., 1], x[..., 2]
    r = sqrt(z**2 + x**2)
    k = r**2 / rc**2

    my = exp(-2 * k)
    mx = -z / r * sqrt(1 - exp(-4 * k))
    mz = x / r * sqrt(1 - exp(-4 * k))

    mag = stack([mx, my, mz], axis=-1)
    return unit_vec(mag)


def sample_uniform_state(key):
    return uniform_state


def sample_vortex_state(key, rc_min=0.1, rc_max=1.0):
    rc = random.uniform(key, (), minval=rc_min, maxval=rc_max)
    return partial(vortex_state, rc=rc)


def sample_flower_state(key, a_min=0.5, a_max=2.0, b_min=1, b_max=3, c_min=0.5, c_max=2):
    a = random.uniform(key, (), minval=a_min, maxval=a_max)
    b = random.uniform(key, (), minval=b_min, maxval=b_max)
    c = random.uniform(key, (), minval=c_min, maxval=c_max)
    return partial(flower_state, a=a, b=b, c=c)


default_init_mags = [sample_uniform_state, sample_vortex_state, sample_flower_state]


def init_mag(x: Array, key: Array, sample_functions: list[Callable] = default_init_mags) -> Array:
    """This function provides a broad range of initial magnetization states.
    Note that the key must be the same for each state.

    Parameters
    ----------
    x : Array
    key : Array
    sample_fn : list[Callable], optional
        list of sample functions for the initial magnetization, by default default_init_mags

    Returns
    -------
    Array
        initial magnetization at x
    """
    k1, k2, k3 = random.split(key, 3)
    index = random.randint(k1, (), minval=0, maxval=len(sample_functions))
    _init_mag_sample = [lambda key: fn(key)(x) for fn in sample_functions]
    m0 = lax.switch(index, _init_mag_sample, k2)
    euler_angles = random.uniform(k3, (3,), minval=-pi, maxval=pi)
    euler_angles = euler_angles.at[1].set(euler_angles[1] / 2)
    q = from_euler_angles(euler_angles)
    m0 = quaternion_rotation(m0, q)
    return m0


@partial(register_dataclass, data_fields=["elm_params", "init_mag_key"], meta_fields=[])
@dataclasses.dataclass
class MagParams:
    elm_params: Array
    init_mag_key: Array


Mag = Callable[..., Array]


def default_mag_elm(x, gamma: float = 2.0, lb: Array = asarray(-0.5), ub: Array = asarray(0.5)):
    with jax.ensure_compile_time_eval():
        c = asarray(PoissonDisk(3, radius=0.1, rng=42).fill_space())
        c = c * (ub - lb) + lb

    return jnp.exp(-gamma * norm(x - c, axis=-1) ** 2)


default_elm_size = default_mag_elm(zeros((3,))).shape[0]


def default_mag_model(
    x, params: MagParams, elm: ELM = default_mag_elm, sample_functions: list[Callable] = default_init_mags
) -> Array:
    m0 = init_mag(x, params.init_mag_key, sample_functions=sample_functions)
    p = elm(x) @ params.elm_params
    assert p.shape == (3,)
    return cayley_rotation(p, m0)


def draw_mag_params(key: Array, elm_size: int = default_elm_size) -> MagParams:
    k1, k2, k3 = random.split(key, 3)
    p = random.normal(k1, (elm_size, 3))
    scale = random.normal(k2, ())
    return MagParams(p * scale, k3)


@partial(
    jit,
    static_argnames=(
        "n",
        "mag_model",
        "mag_params_sample_fn",
    ),
)
def sample_magnetization_states(
    key: Array,
    n: int,
    potential_solver: PotentialSolver,
    mag_model: Mag = default_mag_model,
    mag_params_sample_fn: Callable = draw_mag_params,
    max_exchange_energy=15,
    tol: float = 1e-2,
) -> tuple[MagParams, ElmPoissonSolution, Array]:
    def _sample_mag(key) -> tuple[MagParams, ElmPoissonSolution, Array]:
        mag_params = mag_params_sample_fn(key)
        _mag = lambda x: mag_model(x, mag_params)
        poisson_solution = potential_solver.u1_solution(_mag)
        e_ex = exchange_energy(_mag, 1.0, potential_solver.poisson_solver.quad_rule)
        return mag_params, poisson_solution, e_ex

    def _accept_mag(mag_sample, key):
        _, poisson_solution, e_ex = mag_sample
        return (poisson_solution.strong_residual < tol) & (e_ex < max_exchange_energy)

    return rejection_sampling(key, n, _sample_mag, _accept_mag)


def sample_domain(
    key: Array,
    n_samples: int,
    adf: ADF,
    *args: Any,
    dimension: int = 3,
    lower_bound: Array = asarray(-0.5),
    upper_bound: Array = asarray(0.5),
    eps: float = 0.0,
    curvature_threshold: float | None = 100.0,
    interior: bool = True,
    pdf: PDF | None = None,
) -> Array:
    """Rejection sampling based on the given ADF. This function can sample
    the interior of the domain, as well as the exteriour. Further, thresholds can be
    applied to have a minimum distance from the boundary (`eps`) or to
    avoid sampling in regions with high curvature of the ADF (`curvature_threshold`).
    Additionally, samples can be drawn with a specific probability by providing
    the probability density function `pdf`.

    Parameters
    ----------
    key : Array
    n_samples : int
    adf : ADF
    lower_bound : Array, optional
        lower bound of uniform distribution of the samples, by default -1
    upper_bound : Array, optional
        upper bound of uniform distribution of the samples, by default 1
    eps : float, optional
        minimum distance to the boundary - note that this is not a strict measure since it is measured
        by the ADF, by default 0.0.
    curvature_threshold : float | None, optional
        reject samples with `laplace(adf) > curvature_threshold`, by default 100.0
    interior : bool, optional
        If `True`, the interiour is sampled, otherwise the exterior.
        `lower_bound` and `upper_bound` might need to be adjusted to sample
        the exterior, by default True
    pdf : PDF | None, optional
        Probability density function from which to draw the samples from. This is mainly
        thought to sample the exterior with more samples close to the actual domain, by default None.

    Returns
    -------
    Array
        Samples
    """
    _adf = lambda s, *args: asarray(adf(s, *args))
    lower_bound, upper_bound = asarray(lower_bound), asarray(upper_bound)

    def sample_fn(key):
        sample = random.uniform(key, (dimension,), minval=lower_bound, maxval=upper_bound)
        return sample

    def accept_fn(sample, key):
        if pdf is not None:
            p = random.uniform(key)
            if upper_bound.shape == () and lower_bound.shape == ():
                pdf_scaling = (upper_bound - lower_bound) ** 3
            else:
                pdf_scaling = jnp.prod(upper_bound - lower_bound)
            accept = p < (pdf(sample) * pdf_scaling)
        else:
            accept = True

        _l = _adf(sample, *args)
        valid = jnp.abs(_l) > eps
        if curvature_threshold is not None:
            lap_l = laplace(_adf)(sample, *args)
            valid = valid & (jnp.abs(lap_l) < curvature_threshold)
        if interior:
            return accept & (_l > 0) & valid
        else:
            return accept & (_l <= 0) & valid

    _samples = rejection_sampling(key, n_samples, sample_fn, accept_fn)
    return _samples
