from os import makedirs
from os.path import join as join_path

from pinns.prelude import *
from pinns.domain import Hypercube
from pinns.calc import laplace, divergence
from scipy.stats.qmc import Sobol
from pinns.model import mlp
from pinns.opt import train_nn
from pinns.interpolate import shape_function


def main(
    samples_dom=13,
    samples_bnd=11,
    samples_conv=15,
    number_weights=9,
    weights_scale=4,
    mag_layers=3, 
    mag_units=50, 
    epochs=20000, 
    increase_penalty_after=2000,
    penalty_increse=1.5,
    lr_start=1e-5,
    lr_end=1e-7,
    batch_size=50,
    alpha0=1.,
    output_dir=".",
):
    domain = Hypercube((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))
    x_dom = array(Sobol(3, seed=0).random_base2(samples_dom))
    x_dom = domain.transform(x_dom)
    x_bnd = array(Sobol(2, seed=1).random_base2(samples_bnd))
    x_bnd = domain.transform_bnd(x_bnd)

    _x_bnd = array(Sobol(2, seed=1).random_base2(8))
    _x_bnd = domain.transform_bnd(_x_bnd)
    x1 = [_x_bnd[_x_bnd[:, i] == -0.5, :] for i in range(3)]
    x2 = [_x_bnd[_x_bnd[:, i] == 0.5, :] for i in range(3)]
    shape_fun = shape_function(x1, x2)

    def unit_vec(x):
        return x / norm(x, axis=-1, keepdims=True)

    key, subkey = random.split(random.PRNGKey(42), 2)
    mag_model, params = mlp(subkey, [4] + [mag_units] * mag_layers + [3])
    tx = optax.adam(1e-5)
    init_state = TrainState.create(
        apply_fn=mag_model.apply,
        params=params,
        tx=tx
    )

    lam_dom = (8., 9.)

    def draw_lam(key, n):
        return random.uniform(key, (n, )) * (lam_dom[1] - lam_dom[0]) + lam_dom[0]

    with open("init_state_flower.data", "rb") as state_flower:
        init_state_flower = flax.serialization.from_bytes(init_state, state_flower.read())

    with open("init_state_vortex.data", "rb") as state_vortex:
        init_state_vortex = flax.serialization.from_bytes(init_state, state_vortex.read())

    weights = array(Sobol(4, seed=13).random_base2(number_weights))
    W_elm = (weights[:, :3] * 2 - 1) * weights_scale
    b_elm = (weights[:, 3] * 2 - 1) * weights_scale

    _x_bnd = array(Sobol(2, seed=131).random_base2(samples_conv))
    _x_bnd = domain.transform_bnd(_x_bnd)

    def phi2_solution(x, x_bnd, phi1, shape_fun, m):
        eps = 1e-9

        def g(y):
            n = unit_vec(-grad(shape_fun)(y))
            return dot(m(y), n) - dot(grad(phi1)(y), n)

        dist = vmap(lambda x: norm(x - x_bnd, axis=-1))(x)
        _g = vmap(g)(x_bnd)

        def kernel(dist):
            idx = dist > eps
            newton_kernel = where(idx, 1 / dist, 0.)
            N = jnp.count_nonzero(idx)
            return 6 / (4 * pi * N) * dot(newton_kernel, _g)

        return vmap(kernel)(dist)

    h_elm = lambda x: tanh(W_elm @ x + b_elm)
    u_elm = lambda x: shape_fun(x) * h_elm(x)
    Q_elm = vmap(lambda x: -laplace(u_elm)(x))(x_dom)
    U_elm, S_elm, VT_elm = jax.scipy.linalg.svd(Q_elm, full_matrices=False, lapack_driver="gesvd")
    H_bnd_elm = vmap(h_elm)(x_bnd)
    U_bnd_elm, S_bnd_elm, VT_bnd_elm = jax.scipy.linalg.svd(H_bnd_elm, full_matrices=False, lapack_driver="gesvd")

    def solve_stray_field(m):
        f = lambda x: -divergence(m)(x)
        b1 = vmap(f)(x_dom)
        params_phi1 = VT_elm.T @ ((1 / S_elm) * (U_elm.T @ b1))
        phi1 = lambda x: u_elm(x) @ params_phi1
        phi_bnd = phi2_solution(x_bnd, _x_bnd, phi1, shape_fun, m)
        params_phi_bnd = VT_bnd_elm.T @ ((1 / S_bnd_elm) * (U_bnd_elm.T @ phi_bnd))
        g2 = lambda x: h_elm(x) @ params_phi_bnd
        b2 = vmap(laplace(g2))(x_dom)
        params_phi2 = VT_elm.T @ ((1 / S_elm) * (U_elm.T @ b2))
        phi2 = lambda x: u_elm(x) @ params_phi2 + g2(x)
        phi_nn = lambda x: phi1(x) + phi2(x)
        return phi_nn

    def exchange_energy(m, x, lam):
        A = 1 / (lam ** 2)

        def e_ex(x):
            dm = jacfwd(m)(x)
            return jnp.sum(dm * dm)
        return A * mean(vmap(e_ex)(x))

    def ani_energy(m, x):
        def e_ani(x):
            c = array([0., 0., 1.])
            return 1 - (m(x) @ c) ** 2

        return 0.1 * mean(vmap(e_ani)(x))

    def mag_energy(phi, m, x):
        def e_mag(x):
            h = lambda x: -grad(phi)(x)
            e = lambda x: dot(m(x), h(x))
            return e(x)

        return - mean(vmap(e_mag)(x))

    def norm_con(m, x):
        def con(x):
            _m = m(x)
            return (norm(_m) - 1) ** 2
        return mean(vmap(con)(x))

    @jit
    def loss(params, x, lam, alpha):
        _m = lambda x: unit_vec(mag_model.apply(lax.stop_gradient(params), concatenate((x, lam[None]))))
        m = lambda x: (mag_model.apply(params, concatenate((x, lam[None]))))
        phi = solve_stray_field(_m)

        e_ex = exchange_energy(m, x, lam)
        e_ani = ani_energy(m, x)
        e_mag = mag_energy(phi, m, x)
        con = norm_con(m, x)
        e = e_ex + e_ani + e_mag
        _l = e_ex + e_ani + 2 * e_mag + alpha * con
        return _l, {'loss': _l, 'energy': e,
                    'e_ex': e_ex, 'e_ani': e_ani,
                    'e_mag': e_mag, 'norm_con': con}

    def train_mag(key, init_state,
                  initialize=False,
                  epochs=2000,
                  batch_size=32,
                  alpha=1.,
                  increase_penalty_after=100):
        @jit
        def make_batches(rng):
            batches = x_dom.shape[0] // batch_size
            k1, k2 = random.split(rng)
            batch_size_dom = x_dom.shape[0] // batches

            perms_dom = jax.random.permutation(k1, x_dom.shape[0])
            perms_dom = perms_dom[:batches * batch_size_dom]  # skip incomplete batch
            perms_dom = perms_dom.reshape((batches, batch_size_dom))

            lam = draw_lam(k2, batches)

            return x_dom[perms_dom], lam

        if initialize:
            batches = x_dom.shape[0] // batch_size
            schedule = optax.linear_schedule(lr_start, lr_end, int(epochs * batches / 1.5))
            tx = optax.radam(schedule)

            init_state = TrainState.create(
                apply_fn=mag_model.apply,
                params=init_state.params,
                tx=tx
            )
        cycles = epochs // increase_penalty_after
        cycles = max(1, cycles)
        state = init_state
        hists = []
        for cycle in range(cycles):
            state, hist = train_nn(loss,
                                   state, make_batches, key,
                                   epochs=increase_penalty_after,
                                   alpha=alpha * penalty_increse ** cycle)
            hists.append(hist)
        full_hist = tree_map(
            lambda *args: concatenate(args),
            *hists)
        return state, full_hist

    print("train flower state")
    key, train_key = random.split(key)
    flower_state, hist_flower = train_mag(train_key,
                                          init_state_flower,
                                          initialize=True,
                                          epochs=epochs,
                                          batch_size=batch_size,
                                          alpha=float(alpha0),
                                          increase_penalty_after=increase_penalty_after)
    print("flower state finished")
    print("train vortex state")
    key, train_key = random.split(key)
    vortex_state, hist_vortex = train_mag(train_key,
                                          init_state_vortex,
                                          initialize=True,
                                          epochs=epochs,
                                          batch_size=batch_size,
                                          alpha=float(alpha0),
                                          increase_penalty_after=increase_penalty_after)

    def save(data, filename):
        model_bytes = flax.serialization.to_bytes(data)
        with open(filename, "wb") as f:
            f.write(model_bytes)

    makedirs(output_dir, exist_ok=True)
    save(flower_state, join_path(output_dir, "state_flower.data"))
    save(hist_flower, join_path(output_dir, "hist_flower.data"))
    save(vortex_state, join_path(output_dir, "state_vortex.data"))
    save(hist_vortex, join_path(output_dir, "hist_vortex.data"))


if __name__ == '__main__':
    import fire
    fire.Fire(main)
