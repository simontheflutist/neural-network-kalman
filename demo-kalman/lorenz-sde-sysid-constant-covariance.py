import enum
import itertools
import os
import sys
from enum import Enum
from typing import Tuple

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
from network import *
from normal import *
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
        "num_epochs": 20_000,
        "N": 40_000,
        "R": jnp.eye(1) * 1e-2,
        "opt": optax.adamw(learning_rate=1e-3, nesterov=False),
    },
    LorenzArgs(σ=1e-3, T=0.3, dt=1e-2): {
        "hidden_width": 64,
        "hidden_depth": 4,
        "batch_size": 20_000,
        "num_epochs": 20_000,
        "N": 40_000,
        "R": jnp.eye(1) * 1e-2,
        "opt": optax.adamw(learning_rate=1e-4, nesterov=False),
    },
    # highest T
    LorenzArgs(σ=1e-3, T=1.0, dt=1e-2): {
        "hidden_width": 64,
        "hidden_depth": 5,
        "batch_size": 10_000,
        "num_epochs": 20_000,
        "N": 200_000,
        "R": jnp.eye(1) * 1e-2,
        "opt": optax.adamw(
            learning_rate=optax.piecewise_constant_schedule(1e-4, {200_000: 1e-5}),
            nesterov=False,
        ),
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


def make_data(request: LorenzDataRequest, cache_data=True):
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
    if cache_data and cache_path.exists():
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


def get_model(lorenz_args, cache_model=True, cache_data=True):
    logger.info(f"Generating data for Lorenz system with args {lorenz_args}")
    X_train, Y_train, X_val, Y_val = make_data(
        LorenzDataRequest(
            lorenz_args, jax.random.PRNGKey(42), configuration[lorenz_args]["N"]
        ),
        cache_data=cache_data,
    )
    model_path = get_cache_path(Path(".cache/lorenz_model"), lorenz_args)
    if cache_model and model_path.exists():
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
    logger.info(f"Generating test data for Lorenz system with args {lorenz}")
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


class InferenceKind(Enum):
    PRED = r"Prediction $(t|t-1)$"
    POST = r"Filtering $(t|t)$"
    SMOOTH = r"Smoothing $(t|T)$"
    CONST = r"Constant $(t|\infty)$"


class PerformanceKind(Enum):
    RMSE = r"RMSE"
    COVERAGE95 = r"Coverage at 95\%"
    COVERAGE99 = r"Coverage at 99\%"
    LPDF = r"Log pdf"


class Method(Enum):
    ANALYTIC = enum.auto()
    MEAN_FIELD = enum.auto()
    LINEAR = enum.auto()
    UNSCENTED0 = enum.auto()
    UNSCENTED1 = enum.auto()


class Recalibrate(Enum):
    NO = enum.auto()
    YES = enum.auto()


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

    x_pred: Tuple[Normal]
    x_post: Tuple[Normal]
    x_smooth: Tuple[Normal]
    x_const: Tuple[Normal]

    def __init__(
        self, lorenz_args, test_x, R, uq_params, recalibrate=False, progress_bar=True
    ):
        self.lorenz_args = lorenz_args
        self.test_x = test_x
        self.R = R
        self.uq_params = uq_params
        self.recalibrate = recalibrate

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
            diagnostic_times=slice(1000, -1000),
        )

        sim_horizon = len(self.test_x)
        # generate output
        logger.info("Generating output")
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
        for i in (trange if progress_bar else range)(1, sim_horizon):
            x_and_y_pred = self.kalman.predict(x_post[i - 1].rectify()).rectify()
            x_pred.append(x_and_y_pred[self.kalman.STATES].rectify())
            x_post.append(
                self.kalman.correct(
                    x_and_y_pred, y[i], recalibrate=self.recalibrate
                ).rectify()
            )

        logger.info("Running RTS smoother")
        x_smooth = list(x_post)
        for i in (trange if progress_bar else range)(sim_horizon - 2, -1, -1):
            x_smooth[i] = self.kalman.smooth(x_smooth[i], x_smooth[i + 1]).rectify()

        self.x_pred = tuple(z for z in x_pred)
        self.x_post = tuple(z for z in x_post)
        self.x_smooth = tuple(z for z in x_smooth)
        self.x_const = (Normal.from_samples(self.test_x),) * sim_horizon
        if progress_bar:
            logger.info("Done running Kalman filter and smoother")

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

    def get_inference(self, inference_kind: InferenceKind):
        if inference_kind == InferenceKind.PRED:
            return self.x_pred
        elif inference_kind == InferenceKind.POST:
            return self.x_post
        elif inference_kind == InferenceKind.SMOOTH:
            return self.x_smooth
        elif inference_kind == InferenceKind.CONST:
            return self.x_const
        else:
            raise ValueError(f"Unknown inference kind: {inference_kind}")

    def coverage(self, percentage_point=0.99, inference_kind=InferenceKind.PRED):
        return self.kalman_diagnostics.single_coverage(
            percentage_point, self.get_inference(inference_kind)
        )

    def lpdf(self, inference_kind=InferenceKind.PRED):
        return self.kalman_diagnostics.lpdf(self.get_inference(inference_kind))

    def rmse(self, inference_kind=InferenceKind.PRED):
        return self.kalman_diagnostics.point_rmse(self.get_inference(inference_kind))

    def get_performance(
        self, performance_kind: PerformanceKind, inference_kind: InferenceKind
    ):
        if performance_kind == PerformanceKind.RMSE:
            return self.rmse(inference_kind)
        elif performance_kind == PerformanceKind.COVERAGE95:
            return self.coverage(0.95, inference_kind)
        elif performance_kind == PerformanceKind.COVERAGE99:
            return self.coverage(0.99, inference_kind)
        elif performance_kind == PerformanceKind.LPDF:
            return self.lpdf(inference_kind)
        else:
            raise ValueError(f"Unknown performance kind: {performance_kind}")


def mean_and_se(arr):
    """
    Calculate the mean and standard error of an array of data.

    Parameters
    ----------
    arr : array-like
        The array of data.

    Returns
    -------
    mean : array-like
        The mean of the data.
    se : array-like
        The standard error of the data.
    """
    arr = np.array(list(arr))
    return np.mean(arr, axis=0), np.std(arr, axis=0) / np.sqrt(len(arr))


def get_pretty_kalman_kind(method: Method, recalibrate: Recalibrate):
    return (
        r"{\textsc{"
        + {
            Method.ANALYTIC: "analytic",
            Method.LINEAR: "linear",
            Method.UNSCENTED0: "unscented'95",
            Method.UNSCENTED1: "unscented'02",
            Method.MEAN_FIELD: "mean-field",
        }[method]
        + (" (recal)" if recalibrate == Recalibrate.YES else "")
        + "}}"
    )


def get_pretty_inference_kind(inference_kind: InferenceKind):
    return {
        InferenceKind.PRED: "prediction",
        InferenceKind.POST: "filtering",
        InferenceKind.SMOOTH: "smoothing",
        InferenceKind.CONST: "constant",
    }[inference_kind]


def format_scientific(x, implicit_plus=True):
    if np.isnan(x) or jnp.isnan(x):
        return "---"
    if x == 0:
        return "0"
    return (
        r"""\num[print-zero-exponent = true,print-implicit-plus="""
        + ("true" if implicit_plus else "false")
        + r",print-exponent-implicit-plus=true]{"
        + f"{x:.3e}"
        + "}"
    )


def format_std(x):
    if np.isnan(x) or jnp.isnan(x):
        return "---"
    if x == 0:
        return "0"
    return (
        r"""\num[print-zero-exponent = true,print-exponent-implicit-plus=true]{"""
        + f"{x:.1e}"
        + "}"
    )


def format_scientific_uncertainty(mean, std, implicit_plus=True):
    return (
        format_scientific(mean, implicit_plus=implicit_plus)
        + r" \ensuremath{\pm} "
        + format_std(std)
    )


def results_to_dataframe(performance_kind, inference_kinds):
    """Convert results to a pandas DataFrame for a given performance kind.

    Args:
        performance_kind: The performance metric to extract (e.g., PerformanceKind.RMSE)

    Returns:
        A pandas DataFrame with (method, recalibrate) as rows and inference_kind as columns
    """
    import pandas as pd

    # Create multi-index for rows
    index = pd.MultiIndex.from_product(
        [list(Method), list(Recalibrate)],
        names=["Method", "Recalibrate"],
    )

    # Create DataFrame with pretty-printed inference kinds as columns
    pretty_columns = [
        get_pretty_inference_kind(inf_kind) for inf_kind in inference_kinds
    ]
    df = pd.DataFrame(index=index, columns=pretty_columns, dtype=object)

    # Fill in the DataFrame
    for inf_kind, pretty_col in zip(inference_kinds, pretty_columns):
        for (method, recalibrate), (mean, se) in results[performance_kind][
            inf_kind
        ].items():
            # Format using scientific notation with uncertainty
            df.loc[(method, recalibrate), pretty_col] = format_scientific_uncertainty(
                mean, se
            )

    return df


def const_results_to_latex(filename):
    """Generate a LaTeX table showing CONST inference results for all performance kinds.

    Args:
        filename: Output file path for the LaTeX table
    """
    import pandas as pd

    # Create a list to hold all data
    data = []

    # Collect only the first method for each performance kind
    for perf_kind in PerformanceKind:
        # Get the first (method, recalibrate) pair and its values
        (method, recalibrate), (mean, se) = next(
            iter(results[perf_kind][InferenceKind.CONST].items())
        )
        data.append(
            {
                "Performance": perf_kind.value,
                "Value": format_scientific_uncertainty(mean, se),
            }
        )

    # Create DataFrame
    df = pd.DataFrame(data)

    # Generate LaTeX table with just Performance and Value columns
    df.to_latex(
        buf=filename,
        index=False,
        escape=False,
        column_format="lc",
        caption=r"Performance metrics for \textsc{stationary} inference",
        label="tab:const_results",
        position="htbp",
    )


def results_to_latex(
    filename, performance_kind, inference_kinds, caption=None, label=None
):
    """Convert results to a LaTeX table for a given performance kind.

    Args:
        performance_kind: The performance metric to extract (e.g., PerformanceKind.RMSE)
        caption: Optional caption for the table
        label: Optional label for the table

    Returns:
        A string containing the LaTeX table
    """
    df = results_to_dataframe(performance_kind, inference_kinds)

    # Reset index to make MultiIndex into columns
    df = df.reset_index()

    # Combine Method and Recalibrate into a single column
    df["Method"] = df.apply(
        lambda row: get_pretty_kalman_kind(row["Method"], row["Recalibrate"]),
        axis=1,
    )

    # Drop the separate Recalibrate column
    df = df.drop(columns=["Recalibrate"])

    # Set column format (single column for method, rest for data)
    column_format = "l" + "l" * (len(df.columns) - 1)

    # Generate LaTeX
    df.to_latex(
        buf=filename,
        index=False,
        escape=False,
        column_format=column_format,
        caption=caption
        or f"{performance_kind.value} "
        + "("
        + ", ".join(
            [get_pretty_inference_kind(inf_kind) for inf_kind in inference_kinds]
        )
        + ")",
        label=label
        or f"tab:results_{performance_kind.name.lower()}_{'_'.join([inf_kind.name.lower() for inf_kind in inference_kinds])}",
        position="htbp",
        float_format="%.3g",
    )


def get_kalman_filter_types():
    kalman_filter_types = list(itertools.product(Method, Recalibrate))
    # kalman_filter_types = [
    #     (Method.UNSCENTED0, Recalibrate.NO),
    #     (Method.UNSCENTED0, Recalibrate.YES),
    # ]
    with tqdm(total=len(kalman_filter_types)) as pbar:
        for method, recalibrate in kalman_filter_types:
            pbar.set_description(f"{method}, {recalibrate}")
            yield method, recalibrate
            pbar.update(1)


use_cached_results = True

if __name__ == "__main__":
    if use_cached_results:

        logger.info("Loading cached results")
        import shelve

        with shelve.open("results") as shelf:
            results = shelf["results"]

        const_results_to_latex(
            "../docs/kalman-manuscript/generated/tables/stationary.tex"
        )

        with open("../docs/kalman-manuscript/generated/generated-tables.tex", "w") as f:
            for performance_kind in PerformanceKind:
                rel_path_table_1 = (
                    "generated/tables/"
                    + performance_kind.name.lower()
                    + "-pred-post.tex"
                )
                rel_path_table_2 = (
                    "generated/tables/" + performance_kind.name.lower() + "-smooth.tex"
                )
                results_to_latex(
                    "../docs/kalman-manuscript/" + rel_path_table_1,
                    performance_kind,
                    inference_kinds=[InferenceKind.PRED, InferenceKind.POST],
                    label="",
                )
                results_to_latex(
                    "../docs/kalman-manuscript/" + rel_path_table_2,
                    performance_kind,
                    inference_kinds=[InferenceKind.SMOOTH],
                    label="",
                )
                print(r"\input{" + rel_path_table_1 + "}", file=f)
                print(r"\input{" + rel_path_table_2 + "}", file=f)
        sys.exit(0)

    lorenz_args = LorenzArgs(σ=1e-3, T=1.0, dt=1e-2)
    model, Q = get_model(lorenz_args, cache_model=True, cache_data=True)
    lorenz = LorenzSDE(lorenz_args)

    results = {
        performance_kind: {inference_kind: dict() for inference_kind in InferenceKind}
        for performance_kind in PerformanceKind
    }
    test_datasets = make_test_data(
        lorenz, 10000, jax.random.split(jax.random.PRNGKey(10), 20)
    )

    with open("../docs/kalman-manuscript/generated/generated.tex", "w") as f:
        for method, recalibrate in get_kalman_filter_types():
            test_cases = [
                TestCase(
                    lorenz_args,
                    test_data,
                    configuration[lorenz_args]["R"],
                    {
                        Method.ANALYTIC: dict(
                            method="analytic",
                            rectify=True,
                        ),
                        Method.LINEAR: dict(
                            method="linear",
                            rectify=True,
                        ),
                        Method.UNSCENTED0: dict(
                            method="unscented",
                            unscented_method=unscented.UnscentedTransformMethod.UT0_VECTOR,
                            rectify=True,
                        ),
                        Method.UNSCENTED1: dict(
                            method="unscented",
                            unscented_method=unscented.UnscentedTransformMethod.UT1_VECTOR,
                            rectify=True,
                        ),
                        Method.MEAN_FIELD: dict(
                            method="analytic",
                            mean_field=True,
                            rectify=True,
                        ),
                    }[method],
                    progress_bar=False,
                    recalibrate=True if recalibrate == Recalibrate.YES else False,
                )
                for test_data in test_datasets
            ]

            kalman_kind_pretty_name = get_pretty_kalman_kind(method, recalibrate)

            # save trajectory and coverage plots
            file_name = f"{method}-{recalibrate}"
            rel_path_trajectory = "generated/trajectory/" + file_name + ".pdf"
            rel_path_coverage = "generated/coverage/" + file_name + ".pdf"

            test_cases[0].plot_trajectory(
                Figure(figsize=(8, 6), dpi=600, constrained_layout=True)
            ).savefig("../docs/kalman-manuscript/" + rel_path_trajectory)
            test_cases[0].plot_coverage(
                Figure(figsize=(6, 6), dpi=600, constrained_layout=True)
            ).savefig("../docs/kalman-manuscript/" + rel_path_coverage)

            print(f"\\subsection{{Kalman Filter: {kalman_kind_pretty_name}}}", file=f)
            print(r"\begin{figure}[H]", file=f)
            print(r"\begin{center}", file=f)
            print(
                f"\\includegraphics[width=\\linewidth]{{{rel_path_trajectory}}}", file=f
            )
            print(r"\end{center}", file=f)
            print(
                rf"\caption{{Trajectory excerpt for Kalman filter \textsc{{{kalman_kind_pretty_name}}}}}",
                file=f,
            )
            print(r"\end{figure}", file=f)

            print(r"\begin{figure}[H]", file=f)
            print(r"\begin{center}", file=f)
            print(
                f"\\includegraphics[width=\\linewidth]{{{rel_path_coverage}}}", file=f
            )
            print(r"\end{center}", file=f)
            print(
                rf"\caption{{Coverage for Kalman filter \textsc{{{kalman_kind_pretty_name}}}}}",
                file=f,
            )
            print(r"\end{figure}", file=f)

            f.flush()
            for performance_kind in PerformanceKind:
                for inference_kind in InferenceKind:
                    logger.info(
                        f"Computing {performance_kind} for {inference_kind} with {method} and {recalibrate}"
                    )
                    results[performance_kind][inference_kind][(method, recalibrate)] = (
                        mean_and_se(
                            test_case.get_performance(performance_kind, inference_kind)
                            for test_case in test_cases
                        )
                    )

    logger.info("Saving results to LaTeX tables")

    with open("../docs/kalman-manuscript/generated/generated-tables.tex", "w") as f:
        for performance_kind in PerformanceKind:
            rel_path_table_1 = (
                "generated/tables/" + performance_kind.name.lower() + "-pred-post.tex"
            )
            rel_path_table_2 = (
                "generated/tables/" + performance_kind.name.lower() + "-smooth.tex"
            )
            results_to_latex(
                "../docs/kalman-manuscript/" + rel_path_table_1,
                performance_kind,
                inference_kinds=[InferenceKind.PRED, InferenceKind.POST],
                label="",
            )
            results_to_latex(
                "../docs/kalman-manuscript/" + rel_path_table_2,
                performance_kind,
                inference_kinds=[InferenceKind.SMOOTH],
                label="",
            )
            print(r"\input{" + rel_path_table_1 + "}", file=f)
            print(r"\input{" + rel_path_table_2 + "}", file=f)

    import shelve

    with shelve.open("results") as shelf:
        shelf["results"] = results

    # import IPython

    # IPython.embed(colors="neutral")

    # lorenz_args = LorenzArgs(σ=0.5, T=0.5, dt=1e-2)
    # lorenz = LorenzSDE(lorenz_args)
    # test_datasets = make_test_data(
    #     lorenz, 2000, jax.random.split(jax.random.PRNGKey(10), 10)
    # )
    # test_cases = [
    #     TestCase(
    #         lorenz_args,
    #         test_data,
    #         configuration[lorenz_args]["R"],
    #         dict(
    #             method="linear",
    #             unscented_method=unscented.UnscentedTransformMethod.UT1_VECTOR,
    #             rectify=True,
    #         ),
    #     )
    #     for test_data in test_datasets
    # ]
    # print(mean_and_se([test_case.rmse() for test_case in test_cases]))
