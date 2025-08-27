import os
import sys

lib_path = os.path.join(os.path.curdir, "../src")
sys.path.insert(0, lib_path)


import hashlib
from dataclasses import dataclass
from pathlib import Path

import equinox
import jax
import optax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
import logging

import activation
import jax.numpy as jnp
import kalman_diagnostics
import neural_kalman
import numpy as np
import random_matrix
import unscented
from matplotlib.figure import Figure
from tqdm import tqdm, trange

from lorenz.lorenz import LorenzArgs, LorenzSDE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

configuration = {
    # reproduces github notebook example
    LorenzArgs(σ=0.5, T=0.5, dt=1e-2): {
        "hidden_width": 30,
        "hidden_depth": 3,
        "batch_size": 40_000,
        "num_epochs": 20000,
        "N": 40_000,
        "R": jnp.eye(1) * 1e-2,
        "opt": optax.adamw(learning_rate=1e-3, nesterov=False),
    },
    LorenzArgs(σ=1e-3, T=0.3, dt=1e-2): {
        "hidden_width": 64,
        "hidden_depth": 4,
        "batch_size": 20_000,
        "num_epochs": 20000,
        "N": 40_000,
        "R": jnp.eye(1) * 1e-2,
        "opt": optax.adamw(learning_rate=1e-4, nesterov=False),
    },
    LorenzArgs(σ=1e-3, T=1.0, dt=1e-2): {
        "hidden_width": 64,
        "hidden_depth": 5,
        "batch_size": 10000,
        "num_epochs": 20000,
        "N": 200_000,
        "R": jnp.eye(1) * 1e-2,
        "opt": optax.adamw(learning_rate=1e-4, nesterov=True),
    },
}


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
    x_train_val = np.zeros((request.N + 1, 2, 3))
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

    for i, keys in enumerate(tqdm(jax.random.split(request.key, (request.N, 2)))):
        x_train_val[i + 1, :, :] = F_true_train_val(x_train_val[i, :], keys).squeeze()

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
            A=random_matrix.RandomGaussian(scale=1e0 * X_train.std(0) ** -1),
            b=random_matrix.RandomUniform(),
        ),
        *[
            layer_factory(
                in_size=hidden_width,
                out_size=hidden_width,
                activation=σ,
                key=jax.random.PRNGKey(i + 2),
                A=random_matrix.RandomGaussian(),
                b=random_matrix.RandomUniform(),
            )
            for i in range(hidden_depth - 1)
        ],
        Layer.create_linear(
            in_size=hidden_width,
            out_size=n_x,
            C=random_matrix.ZeroMatrix(),
            d=random_matrix.ConstantMatrix(jnp.mean(X_train, 0)),
        ),
    )

    return network


def train_network(
    network,
    X_train,
    Y_train,
    X_val,
    Y_val,
    opt,
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
        # return jnp.linalg.det(Q) ** (1 / Q.shape[0]), Q

    loss_value_and_grad = equinox.filter_value_and_grad(get_loss, has_aux=True)

    # --- make minibatches ---
    N = X_train.shape[0]
    num_batches = N // batch_size
    logger.info(f"Number of training samples: {N}")
    logger.info(f"Num batches: {num_batches}")

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

    loss, Q, network, opt_state, val_loss, val_Q = step_batch(network, opt_state)
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


def get_model(lorenz_args):
    logger.info(f"Generating data for Lorenz system with args {lorenz_args}")
    X_train, Y_train, X_val, Y_val = make_data(
        LorenzDataRequest(
            lorenz_args, jax.random.PRNGKey(42), configuration[lorenz_args]["N"]
        )
    )
    model_path = get_cache_path(Path(".cache/lorenz_model"), lorenz_args)
    if model_path.exists():
        logger.info(f"Loading model from cache: {model_path}")
        skeleton = initialize_network(
            X_train,
            hidden_width=configuration[lorenz_args]["hidden_width"],
            hidden_depth=configuration[lorenz_args]["hidden_depth"],
        )
        return equinox.tree_deserialise_leaves(
            model_path, (skeleton, jnp.zeros((3, 3)))
        )
    else:
        logger.info(f"Training model with args {lorenz_args}")
        network, Q = train_network(
            initialize_network(
                X_train,
                hidden_width=configuration[lorenz_args]["hidden_width"],
                hidden_depth=configuration[lorenz_args]["hidden_depth"],
            ),
            X_train,
            Y_train,
            X_val,
            Y_val,
            opt=configuration[lorenz_args]["opt"],
            batch_size=configuration[lorenz_args]["batch_size"],
            num_epochs=configuration[lorenz_args]["num_epochs"],
        )
        logger.info(f"Saving model to cache: {model_path}")
        equinox.tree_serialise_leaves(model_path, (network, Q))
        return network, Q


def make_test_data(lorenz, N, trial_keys, x_0=np.array([-8.0, 4.0, 27.0])):
    """
    Generate test data.

    Args:
        lorenz: an instance of LorenzSDE
        N: the number of time steps
        trial_keys: a sequence of keys for random number generation

    Returns:
        A jax.Array of shape (num_trials, N + 1, 3) containing the test data
    """

    num_trials = len(trial_keys)
    noise_keys = jax.vmap(jax.random.split, in_axes=(0, None), out_axes=1)(
        trial_keys, N
    )
    x = np.zeros((N + 1, num_trials, 3))
    x[0, :, :] = x_0
    _F = jax.jit(jax.vmap(lorenz.F))
    for i, keys in enumerate(tqdm(noise_keys)):
        x[i + 1, :, :] = _F(x[i, :, :], keys).squeeze()
    return jnp.transpose(x, (1, 0, 2))


class TestCase(equinox.Module):
    # passed in by user
    lorenz_args: LorenzArgs
    test_x: jax.Array
    R: jax.Array
    uq_params: dict
    key = jax.random.PRNGKey(12324567890)
    recalibrate: bool = False

    # RAII
    lorenz: LorenzSDE
    model: Network
    Q: jax.Array
    H: jax.Array = Network(
        Layer.create_linear(
            in_size=3,
            out_size=1,
            C=np.eye(3)[(0,), :],
            d=np.zeros((1)),
        )
    )
    kalman: neural_kalman.NeuralKalmanFilter
    kalman_diagnostics: kalman_diagnostics.KalmanDiagnostics

    x_pred: list
    x_post: list
    x_smooth: list
    x_const: list

    def __init__(self, lorenz_args, test_x, R, uq_params):
        self.lorenz_args = lorenz_args
        self.test_x = test_x
        self.R = R
        self.uq_params = uq_params

        self.lorenz = LorenzSDE(lorenz_args)
        self.model, self.Q = get_model(lorenz_args)
        self.kalman = neural_kalman.NeuralKalmanFilter(
            n_x=3,
            n_u=0,
            n_y=1,
            F=self.model,
            H=self.H,
            Q=self.Q,
            R=self.R,
            uq_params=self.uq_params,
        )
        self.kalman_diagnostics = kalman_diagnostics.KalmanDiagnostics(
            kalman_filter=self.kalman,
            x=self.test_x,
            diagnostic_times=slice(10, -10),
        )

        sim_horizon = len(self.test_x)
        # generate output
        y = jax.vmap(self.H)(self.test_x) + jax.random.multivariate_normal(
            mean=jnp.zeros(1),
            cov=self.R,
            key=jax.random.PRNGKey(302),
            shape=sim_horizon,
        )
        # run the Kalman filters
        logger.info("Running Kalman filter")
        x_0 = Normal.certain(self.test_x[0])
        x_pred = [x_0]
        x_post = [x_0]
        for i in trange(1, sim_horizon):
            x_and_y_pred = self.kalman.predict(x_post[i - 1].rectify())
            x_pred.append(x_and_y_pred[self.kalman.STATES])
            x_post.append(
                self.kalman.correct(
                    x_and_y_pred, y[i], recalibrate=self.recalibrate
                ).rectify()
            )

        logger.info("Running RTS smoother")
        x_smooth = list(x_post)
        for i in trange(sim_horizon - 2, -1, -1):
            x_smooth[i] = self.kalman.smooth(x_smooth[i], x_smooth[i + 1]).rectify()

        self.x_pred = x_pred
        self.x_post = x_post
        self.x_smooth = x_smooth
        self.x_const = [Normal.from_samples(self.test_x)] * sim_horizon

    def plot_trajectory(self, fig: Figure):
        time_slice = slice(len(self.x_pred) // 2 - 50, len(self.x_pred) // 2 + 50)
        for i in range(3):
            for j, (x_trajectory, desc) in enumerate(
                [
                    (self.x_pred, "Prediction $(t|t-1)$"),
                    (self.x_post, "Filtering $(t|t)$"),
                    (self.x_smooth, "Smoothing $(t|T)$"),
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
                    self.kalman_diagnostics.plot_state_trajectories(
                        x_trajectory, i, time_slice, coverage=0.90
                    )
                )
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
        return fig

    def plot_coverage(self, fig: Figure):
        ax = fig.add_subplot()
        ax.plot([0, 1], [0, 1], alpha=0)
        ax.plot(
            *self.kalman_diagnostics.calculate_coverage(self.x_pred), label="prediction"
        )
        ax.plot(
            *self.kalman_diagnostics.calculate_coverage(self.x_post), label="filtering"
        )
        ax.plot(
            *self.kalman_diagnostics.calculate_coverage(self.x_smooth),
            label="smoothing",
            linestyle="--",
        )
        ax.plot(
            *self.kalman_diagnostics.calculate_coverage(self.x_const),
            label="stationary",
            linestyle="-",
        )
        ax.set_xlabel("Nominal coverage")
        ax.set_ylabel("Actual coverage")
        ax.legend()
        return fig


if __name__ == "__main__":
    lorenz_args = LorenzArgs(σ=0.5, T=0.5, dt=1e-2)
    lorenz = LorenzSDE(lorenz_args)
    test_data = make_test_data(lorenz, 200, jax.random.split(jax.random.PRNGKey(10), 1))
    test_case = TestCase(
        lorenz_args,
        test_data[0],
        configuration[lorenz_args]["R"],
        dict(
            method="linear",
            unscented_method=unscented.UnscentedTransformMethod.UT1_VECTOR,
            rectify=True,
        ),
    )
    # data_request = LorenzDataRequest(
    #     lorenz_args=lorenz_args, key=jax.random.PRNGKey(23), N=200_000
    # )
    # X_train, Y_train, X_val, Y_val = make_data(data_request)

    # network, Q = train_network(
    #     initialize_network(
    #         X_train,
    #         hidden_width=configuration[lorenz_args]["hidden_width"],
    #         hidden_depth=configuration[lorenz_args]["hidden_depth"],
    #     ),
    #     X_train,
    #     Y_train,
    #     X_val,
    #     Y_val,
    #     batch_size=configuration[lorenz_args]["batch_size"],
    #     num_epochs=configuration[lorenz_args]["num_epochs"],
    # )

    import IPython

    IPython.embed(colors="neutral")
