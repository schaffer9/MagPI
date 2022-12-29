from pinns.prelude import *
from pinns.domain import Hypercube
from scipy.stats.qmc import Sobol
from pinns.model import mlp
from pinns.opt import train_nn


def main(epochs=2000, batch_size=50, samples_dom=13, lr=1e-4, mag_layers=3, mag_units=50):
    domain = Hypercube((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))
    x_dom = array(Sobol(3, seed=0).random_base2(samples_dom))
    x_dom = domain.transform(x_dom)

    def unit_vec(x):
        return x / norm(x, axis=-1, keepdims=True)

    def m_init_vortex(x):
        x, y, z = x[..., 0], x[..., 1], x[..., 2]
        rc = 0.14
        r = sqrt(z ** 2 + x ** 2)
        k = r**2 / rc**2

        my = exp(-2 * k)
        mx = - z / r * sqrt(1 - exp(-4 * k))
        mz = x / r * sqrt(1 - exp(-4 * k))

        mag = stack([mx, my, mz], axis=-1)
        return unit_vec(mag)

    def m_init_flower(x):
        mx = x[..., 0] * x[..., 2]
        my = x[..., 1] * x[..., 2]
        mz = ones_like(my)
        mag = stack([mx, my, mz], axis=-1)
        return unit_vec(mag)

    key, subkey = random.split(random.PRNGKey(42), 2)
    mag_model, params = mlp(subkey, [4] + [mag_units] * mag_layers + [3])
    tx = optax.adam(lr)
    init_state = TrainState.create(
        apply_fn=mag_model.apply,
        params=params,
        tx=tx
    )

    lam_dom = (8., 9.)

    def draw_lam(key, n):
        return random.uniform(key, (n, )) * (lam_dom[1] - lam_dom[0]) + lam_dom[0]

    def train_init_mag(theta_nn, state, init_mag, key, epochs=500, batch_size=100):
        mag = init_mag(x_dom)

        @jit
        def make_batches(rng):
            k1, k2 = random.split(rng)
            batches = x_dom.shape[0] // batch_size

            perms_dom = jax.random.permutation(k1, x_dom.shape[0])
            perms_dom = perms_dom[:batches * batch_size]  # skip incomplete batch
            perms_dom = perms_dom.reshape((batches, batch_size))
            lam = draw_lam(k2, batches)
            return x_dom[perms_dom], mag[perms_dom], lam

        @jit
        def loss(params, x, m, lam):
            lam = ones((x.shape[0])) * lam
            m_pred = theta_nn.apply(params, jnp.append(x, lam[:, None], axis=-1))
            l = mean(norm(m_pred - m, axis=-1) ** 2)
            return l, {'loss': l}

        return train_nn(loss, state, make_batches, key, epochs=epochs)

    key, train_key = random.split(key)
    init_state_flower, hist_flower_init = train_init_mag(
        mag_model,
        init_state,
        m_init_flower,
        train_key, 
        epochs=epochs, 
        batch_size=batch_size)
    key, train_key = random.split(key)
    init_state_vortex, hist_vortex_init = train_init_mag(
        mag_model,
        init_state,
        m_init_vortex,
        train_key,
        epochs=epochs,
        batch_size=batch_size)

    def save(data, filename):
        model_bytes = flax.serialization.to_bytes(data)
        with open(filename, "wb") as f:
            f.write(model_bytes)

    save(init_state_flower, "init_state_flower.data")
    save(hist_flower_init, "hist_flower_init.data")
    save(init_state_vortex, "init_state_vortex.data")
    save(hist_vortex_init, "hist_vortex_init.data")


if __name__ == '__main__':
    import fire
    fire.Fire(main)
