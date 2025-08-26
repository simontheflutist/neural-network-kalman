import os
import sys

lib_path = os.path.join(os.path.curdir, "../src")
sys.path.insert(0, lib_path)

import functools
from dataclasses import dataclass
from pathlib import Path
import hashlib

import diffrax
import equinox
import jax
import optax
import scipy.stats

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
import activation
import jax.numpy as jnp
import kalman_diagnostics
import network
import neural_kalman
import normal
import numpy as np
import random_matrix
import unscented
from matplotlib.figure import Figure
from network import Layer, Network
from normal import Normal
from random_matrix import (
    ConstantMatrix,
    RandomGaussian,
    RandomOrthogonalProjection,
    RandomUniform,
    ZeroMatrix,
)
from tqdm import tqdm, trange

from lorenz.lorenz import LorenzArgs, LorenzSDE


# Set up logging
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LorenzDataRequest:
    """A request for Lorenz system data generation.

    This dataclass encapsulates all parameters needed to generate or retrieve
    cached Lorenz system data.
    """

    lorenz_args: LorenzArgs
    key: jax.random.PRNGKey
    N: int

    def __hash__(self):
        # Create a unique string representation of the dataclass
        data_str = f"{self.lorenz_args.T}_{self.lorenz_args.dt}_{self.N}"
        # Return the hash of this string
        return int(hashlib.sha256(data_str.encode()).hexdigest(), 16) % (2**32)


def get_cache_path(cache_dir: Path, request: LorenzDataRequest) -> Path:
    """Get the cache file path for a given request."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{hash(request)}.eqx"


def make_data(request: LorenzDataRequest):
    """Generate or load cached data for the Lorenz system.

    Args:
        request: A LorenzDataRequest instance containing all parameters needed
                for data generation.

    Returns:
        Tuple containing (X_train, Y_train, X_val, Y_val) as jax.numpy arrays.
    """

    # Generate new data if not in cache
    logger.info(f"Generating data for Lorenz system with args {request.lorenz_args}")

    lorenz = LorenzSDE(request.lorenz_args)
    x_train_val = np.zeros((request.N + 1, 3, 3))
    x_train_val[0, :, :] = [-8.0, 4.0, 27.0]
    F_true_train_val = jax.jit(jax.vmap(lorenz.F))

    X = slice(None, -1)
    Y = slice(1, None)

    X_train = jnp.array(x_train_val[X, 0, :])
    Y_train = jnp.array(x_train_val[Y, 0, :])
    X_val = jnp.array(x_train_val[X, 1, :])
    Y_val = jnp.array(x_train_val[Y, 1, :])

    cache_path = get_cache_path(cache_dir=Path(".cache/lorenz_data"), request=request)

    # Try to load from cache
    if cache_path.exists():
        logger.info(f"Loading data from cache: {cache_path}")
        return equinox.tree_deserialise_leaves(
            cache_path, (X_train, Y_train, X_val, Y_val)
        )

    F_true_train_val = jax.jit(jax.vmap(lorenz.F))

    for i, keys in enumerate(tqdm(jax.random.split(request.key, (request.N, 3)))):
        x_train_val[i + 1, :2, :] = F_true_train_val(
            x_train_val[i, :2], keys[:2]
        ).squeeze()

    X_train = jnp.array(x_train_val[X, 0, :])
    Y_train = jnp.array(x_train_val[Y, 0, :])
    X_val = jnp.array(x_train_val[X, 1, :])
    Y_val = jnp.array(x_train_val[Y, 1, :])
    # Save to cache
    result = (X_train, Y_train, X_val, Y_val)
    equinox.tree_serialise_leaves(cache_path, result)
    logger.info(f"Cached data to {cache_path}")

    return result


def initialize_network(X_train, hidden_width=60, hidden_depth=5):
    layer_factory = Layer.create_residual
    σ = activation.Sinusoid()

    n_x = 3
    network = Network(
        Layer.create_nonlinear(
            in_size=n_x,
            out_size=hidden_width,
            activation=σ,
            key=jax.random.PRNGKey(1312),
            A=RandomGaussian(scale=1e0 * X_train.std(0) ** -1),
            b=RandomUniform(),
        ),
        *[
            layer_factory(
                in_size=hidden_width,
                out_size=hidden_width,
                activation=σ,
                key=jax.random.PRNGKey(i + 2),
                A=RandomGaussian(),
                b=RandomUniform(),
            )
            for i in range(hidden_depth - 1)
        ],
        Layer.create_linear(
            in_size=hidden_width,
            out_size=n_x,
            C=ZeroMatrix(),
            d=ConstantMatrix(jnp.mean(X_train, 0)),
        ),
    )

    return network


def train_network(
    network,
    X_train,
    Y_train,
    X_val,
    Y_val,
    batch_size=10000,
    num_epochs=20000,
):
    @equinox.filter_jit
    def get_loss(model, x, y):
        pred_x = jax.vmap(model)(x)
        actual_x = y
        residual = pred_x - actual_x
        Q = jax.vmap(jnp.outer)(residual, residual).mean(0)
        return jnp.linalg.slogdet(Q, method="qr")[1], Q

    loss_value_and_grad = equinox.filter_value_and_grad(get_loss, has_aux=True)

    # --- make minibatches ---
    N = X_train.shape[0]
    num_batches = N // batch_size
    logger.info(f"Number of training samples: {N}")
    logger.info(f"Num batches: {num_batches}")

    opt = optax.adam(learning_rate=1e-4, nesterov=True)
    opt_state = opt.init(network)

    loss_history = []

    @equinox.filter_jit
    def step_batch(model, opt_state):
        (loss, Q), grads = loss_value_and_grad(model, X_train, Y_train)
        updates, opt_state = opt.update(grads, opt_state, params=model)
        model = equinox.apply_updates(model, updates)
        return loss, Q, model, opt_state, *get_loss(model, X_val, Y_val)

    @equinox.filter_jit
    def step(model, opt_state, key):

        # shuffle indices
        perm = jax.random.permutation(key, N)
        xb = jnp.reshape(perm[: num_batches * batch_size], (num_batches, batch_size))

        def batch_step(carry, idxs):
            model, opt_state, total_loss, total_Q = carry
            x0, x1 = X_train[idxs], Y_train[idxs]

            (loss, Q), grads = loss_value_and_grad(model, x0, x1)
            updates, opt_state = opt.update(grads, opt_state, params=model)
            model = equinox.apply_updates(model, updates)

            # accumulate
            total_loss = total_loss + loss
            total_Q = total_Q + Q
            return (model, opt_state, total_loss, total_Q), None

        # we need an initial Q accumulator with correct shape
        # run one dummy batch to get shape of Q
        (dummy_loss, dummy_Q), _ = loss_value_and_grad(
            model, X_train[:batch_size], Y_train[:batch_size]
        )
        init_Q = jnp.zeros_like(dummy_Q)

        init_carry = (model, opt_state, 0.0, init_Q)
        (model, opt_state, total_loss, total_Q), _ = jax.lax.scan(
            batch_step, init_carry, xb
        )

        # average minibatch results
        avg_loss = total_loss / num_batches
        avg_Q = total_Q / num_batches

        val_loss, val_Q = get_loss(model, X_val, Y_val)
        return avg_loss, avg_Q, model, opt_state, val_loss, val_Q

    loss, Q, _, opt_state, val_loss, val_Q = step_batch(network, opt_state)
    pbar = trange(num_epochs)
    # optional GPU acceleration (not much faster but relieves CPU workload)
    with jax.default_device(jax.devices("cpu")[0]):
        try:
            for i in pbar:
                # loss, Q, network, opt_state, val_loss, val_Q = step_batch(network, opt_state)
                loss, Q, network, opt_state, val_loss, val_Q = step(
                    network,
                    opt_state,
                    key=jax.random.PRNGKey(42 + i),
                )
                # test_loss, test_Q = get_loss(network, x_test[:-1], x_test[1:])
                loss_history.append((loss, val_loss))
                pbar.set_postfix(
                    {
                        "nll": f"{loss:.6f}, {val_loss:.6f}",
                        "rmse": f"{Q.trace()**0.5 :.6f}, {val_Q.trace()**0.5 :.6f}",
                    }
                )
                if i % 20 == 0:
                    fig = Figure(figsize=(4, 2), constrained_layout=1, dpi=300)
                    ax = fig.gca()
                    loss_history_numpy, val_loss_history_numpy = np.array(
                        loss_history
                    ).T
                    ax.set_title("Training")
                    ax.set_ylabel("NLL")
                    # ax.set_yscale("log")
                    ax.set_xlabel("epoch")
                    ax.plot(loss_history_numpy[1:], linewidth=1)
                    val_nll_colors = [
                        "C1" if d > 0 else "C2"
                        for d in val_loss_history_numpy[1:]
                        - val_loss_history_numpy[:-1]
                    ]
                    for i in range(1, len(val_nll_colors)):
                        ax.plot(
                            [i - 1, i],
                            [
                                val_loss_history_numpy[i - 1],
                                val_loss_history_numpy[i],
                            ],
                            color=val_nll_colors[i - 1],
                            linewidth=1,
                        )
                    fig.savefig("/tmp/training.pdf")
        except KeyboardInterrupt:
            logger.info("Training interrupted")
            pass

    return network, Q


if __name__ == "__main__":
    # lorenz_args = LorenzArgs(T=0.1, dt=1e-2)
    lorenz_args = LorenzArgs(T=1.0, dt=1e-2)
    data_request = LorenzDataRequest(
        lorenz_args=lorenz_args, key=jax.random.PRNGKey(23), N=200_000
    )
    X_train, Y_train, X_val, Y_val = make_data(data_request)

    hidden_width = 64
    hidden_depth = 5

    network, Q = train_network(
        initialize_network(
            X_train, hidden_width=hidden_width, hidden_depth=hidden_depth
        ),
        X_train,
        Y_train,
        X_val,
        Y_val,
        batch_size=10000,
        num_epochs=20000,
    )

    equinox.tree_serialise_leaves(
        get_cache_path(Path(".cache/lorenz_model"), data_request), (network, Q)
    )

    import IPython

    IPython.embed(colors="neutral")

raise ValueError("Should not reach here")
# x_pred = np.zeros((10000, 3))
# x_pred[0] = x_train[0]
# for i in trange(1, len(x_pred)):
#     x_pred[i, :] = network(x_pred[i - 1, :])


# fig = Figure(figsize=(8, 4), dpi=100, constrained_layout=1)
# ax = fig.gca()
# ax.plot(np.arange(-200, 0), x_pred[-200:])
# ax.set_xlabel("t")
# fig


x_0 = Normal(x_train[0], val_Q)
x_0 = x_post[-100]


# In[554]:


input_samples = x_0.qmc(2**16)


# In[555]:


true_outputs = jax.vmap(F_true)(
    input_samples, jax.random.split(jax.random.PRNGKey(-432423), len(input_samples))
).squeeze()


# In[556]:


network_outputs = jax.vmap(network)(input_samples)


# In[557]:


dist_unscented = network(x_0, method="unscented")
dist_linear = network(x_0, method="linear")
dist_analytic = network(x_0, method="analytic")


# In[558]:


dist_network = Normal.from_samples(network_outputs)
dist_true = Normal.from_samples(true_outputs)


# In[559]:


print(dist_true)
print(dist_network)


# In[560]:


dist_true.kl_divergence(dist_analytic), dist_network.kl_divergence(dist_analytic)


# In[561]:


dist_true.kl_divergence(dist_unscented), dist_network.kl_divergence(dist_unscented)


# In[562]:


fig = Figure(figsize=(12, 8), dpi=100, constrained_layout=1)
for i in range(3):
    ax = fig.add_subplot(2, 2, i + 1)
    grid = np.linspace(
        np.min(network_outputs[:, i]), np.max(network_outputs[:, i]), 2000
    )
    # ax.plot(
    #     grid,
    #     scipy.stats.gaussian_kde(output_samples[:, i].reshape(-1))(grid),
    #     label="empirical KDE",
    # )
    ax.hist(
        network_outputs[:, i].reshape(-1),
        label="empirical",
        alpha=0.5,
        density=1,
        bins=50,
    )
    for label, dist, linestyle in (
        ("pseudo-true Gaussian fit", dist_network, "-"),
        ("unscented", dist_unscented, "-"),
        ("linear", dist_linear, "-"),
        ("analytic", dist_analytic, "--"),
        ("pseudo-GT", dist_true, "-"),
    ):
        ax.plot(
            grid,
            jax.vmap(dist[i].pdf)(grid),
            label=label,
            linestyle=linestyle,
        )
    ax.legend()

fig


# # Kalman filter example
# ## Generate data

# In[448]:


n_x = 3
n_u = 0
n_y = 1


# In[479]:


sim_horizon = 10000
R = jnp.eye(n_y) * 1e-2


# In[480]:


process_noise_keys = jax.random.split(jax.random.PRNGKey(122), sim_horizon)


# In[481]:


with jax.default_device(jax.devices("cpu")[0]):
    ϵ = jax.random.multivariate_normal(
        mean=jnp.zeros(n_y), cov=R, key=jax.random.PRNGKey(301), shape=sim_horizon
    )


# In[482]:


x = np.ones((sim_horizon, n_x))
# x[0] = [-8.0, 4.0, 27.0]
x[0] = x_val[-1]
y = np.zeros((sim_horizon, n_y))
y_noiseless = np.zeros((sim_horizon, n_y))
for i in trange(1, sim_horizon):
    x[i, :] = F_true(x[i - 1, :], process_noise_keys[i - 1])
    y_noiseless[i, :] = H(x[i, :])
    y[i, :] = y_noiseless[i, :] + ϵ[i]


# Actual state and output trajectory

# In[483]:


time_slice = slice(None, 100)


# In[484]:


fig = Figure(figsize=(8, 6), dpi=100, constrained_layout=1)
ax = fig.add_subplot(211)
for i in range(n_x):
    ax.plot(x[time_slice, i], label=rf"$x_{i+1}$")
ax.legend()
ax = fig.add_subplot(212)
for i in range(n_y):
    ax.plot(y[time_slice, i], label=rf"$y_{i+1}$")
ax.legend()
fig


# ## Filtering

# In[563]:


F_true(x[0], jax.random.PRNGKey(4))


# In[565]:


F = network
H = Network(
    Layer.create_linear(
        in_size=n_x + n_u,
        out_size=n_y,
        C=np.eye(n_x)[(0,), :],
        d=np.zeros((n_y)),
    )
)


# In[566]:


method = "analytic"
# method = "linear"
# method = "unscented"

unscented_method = unscented.UnscentedTransformMethod.UT0_VECTOR
recalibrate = False
rectify = True


# In[567]:


# kalman = neural_kalman.NeuralKalmanFilter(n_x=n_x, n_u=n_u, n_y=n_y, F=F, H=H, Q=Q, R=R)
kalman = neural_kalman.NeuralKalmanFilter(
    n_x=n_x, n_u=n_u, n_y=n_y, F=F, H=H, Q=val_Q, R=R
)


# In[568]:


x_0 = Normal.certain(x[0])
# x_0 = Normal.from_samples(x_train)
x_pred = [x_0]
x_post = [x_0]


# In[569]:


for i in trange(1, sim_horizon):
    x_and_y_pred = kalman.predict(
        x_post[i - 1], method=method, unscented_method=unscented_method, rectify=rectify
    )
    x_pred.append(x_and_y_pred[kalman.STATES])
    x_post.append(
        kalman.correct(
            x_and_y_pred,
            y[i],
            unscented_method=unscented_method,
            recalibrate=recalibrate,
            rectify=rectify,
        )
    )

x_smooth = list(x_post)
for i in trange(sim_horizon - 2, -1, -1):
    x_smooth[i] = kalman.smooth(
        x_smooth[i],
        x_smooth[i + 1],
        method=method,
        unscented_method=unscented_method,
        rectify=rectify,
    )


# In[570]:


x_const = [Normal.from_samples(x_train)] * sim_horizon


# In[571]:


diagnostics = kalman_diagnostics.KalmanDiagnostics(
    kalman, x, diagnostic_times=slice(1000, -1000)
)


# In[572]:


volumes_post = [np.linalg.det(x.Σ) ** 0.5 for x in x_post]


# In[573]:


volumes_smooth = [np.linalg.det(x.Σ) ** 0.5 for x in x_smooth]


# In[574]:


np.mean(volumes_post)


# In[575]:


np.mean(volumes_smooth)


# In[576]:


print(
    f"{'determinant error(predicted x - true x)':<25}",
    diagnostics.point_geometric_error(x_pred),
)
print(
    f"{'determinant error(filtered x - true x)':<25}",
    diagnostics.point_geometric_error(x_post),
)
print(
    f"{'determinant error(smooth x - true x)':<25}",
    diagnostics.point_geometric_error(x_smooth),
)
print(
    f"{'determinant error(const x - true x)':<25}",
    diagnostics.point_geometric_error(x_const),
)


# In[577]:


print(f"{'rms(predicted x - true x)':<25}", diagnostics.point_rmse(x_pred))
print(f"{'rms(filtered x - true x)':<25}", diagnostics.point_rmse(x_post))
print(f"{'rms(smooth x - true x)':<25}", diagnostics.point_rmse(x_smooth))
print(f"{'rms(const x - true x)':<25}", diagnostics.point_rmse(x_const))


# In[497]:


time_slice = slice(sim_horizon // 2 - 50, sim_horizon // 2 + 50)
# time_slice = slice(None, 20)
# time_slice = slice(-50, None)


# Plot the actual states under the 90% prediction interval. Expect 10 misses in this time segment.

# In[498]:


fig = Figure(figsize=(12, 6), dpi=600, constrained_layout=1)
for i in range(n_x):
    for j, (x_trajectory, desc) in enumerate(
        [
            (x_pred, "Prediction $(t|t-1)$"),
            (x_post, "Filtering $(t|t)$"),
            (x_smooth, "Smoothing $(t|T)$"),
        ]
    ):
        # prepare and decorate axes
        ax = fig.add_subplot(3, 3, 3 * i + j + 1)
        if j == 0:
            ax.set_ylabel(rf"$x_{i+1}$")
        else:
            pass  # ax.set_yticks([])
        if i == 0:
            ax.set_title(desc)
        if i == 2:
            ax.set_xlabel(rf"$t$")
        else:
            ax.set_xticks([])
        # get data to plot
        times, state, _mean, lower, upper, covered, missed = (
            diagnostics.plot_state_trajectories(
                x_trajectory, i, time_slice, coverage=0.90
            )
        )
        # plot data
        ax.set_ylim(min(state) - 5, max(state) + 5)
        ax.scatter(
            times[covered],
            state[covered],
            color="C0",
            marker="+",
        )
        ax.scatter(
            times[missed],
            state[missed],
            color="C1",
            marker="x",
        )
        ax.fill_between(
            times,
            lower,
            upper,
            color="C2",
            alpha=0.5,
        )
fig.savefig(f"figures/{method}-trajectory.pdf")
fig


# In[499]:


fig = Figure(dpi=600, constrained_layout=1)
ax = fig.add_subplot()
ax.plot([0, 1], [0, 1], alpha=0)
ax.plot(*diagnostics.calculate_coverage(x_pred), label="prediction")
ax.plot(*diagnostics.calculate_coverage(x_post), label="filtering")
ax.plot(*diagnostics.calculate_coverage(x_smooth), label="smoothing", linestyle="--")
ax.plot(*diagnostics.calculate_coverage(x_const), label="trivial", linestyle="-.")
ax.set_xlabel("Nominal coverage")
ax.set_ylabel("Actual coverage")
ax.legend()
fig.savefig(f"figures/{method}-coverage.pdf")
# fig.savefig(f"/tmp/coverage-{method}.jpg")
# fig.savefig(f"figures/kalman/coverage-{method}.pdf")
fig


# # Performance analysis: I/Os and flops
# ## predict

# In[ ]:


print(f"{'Method':<10} {'I/Os':>10} {'FLOPs':>10}")
for method in ["analytic", "linear", "unscented"]:
    cost_analysis = (
        jax.jit(functools.partial(kalman.predict, method=method))
        .trace(x_post[0])
        .lower()
        .compile()
        .cost_analysis()
    )

    print(
        f"{method:<10} {cost_analysis['bytes accessed']:>10} {cost_analysis['flops']:>10}"
    )


# ## update

# In[ ]:


print(f"{'Method':<10} {"Recalibrate":<10} {'I/Os':>10} {'FLOPs':>10}")
for recalibrate in [True, False]:
    for method in ["analytic", "linear", "unscented"] if recalibrate else ["all"]:
        cost_analysis = (
            jax.jit(
                functools.partial(
                    kalman.correct, recalibrate_method=method, recalibrate=recalibrate
                )
            )
            .trace(
                x_and_y_pred,
                y[0],
            )
            .lower()
            .compile()
            .cost_analysis()
        )
        print(
            f"{method:<10} {str(recalibrate):<10} {cost_analysis['bytes accessed']:>10} {cost_analysis['flops']:>10}"
        )


# ## smooth

# In[ ]:


print(f"{'Method':<10} {'I/Os':>10} {'FLOPs':>10}")
for method in ["analytic", "linear", "unscented"]:
    cost_analysis = (
        jax.jit(functools.partial(kalman.smooth, method=method))
        .trace(x_smooth[0], x_smooth[1])
        .lower()
        .compile()
        .cost_analysis()
    )
    print(
        f"{method:<10} {cost_analysis['bytes accessed']:>10} {cost_analysis['flops']:>10}"
    )


# In[ ]:


# In[ ]:
