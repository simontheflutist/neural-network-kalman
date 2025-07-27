import logging
import sys

import equinox
import jax
import matplotlib
import numpy as np
import optax
import pandas as pd
import scipy.stats
from jax import numpy as jnp

matplotlib.rcParams.update({"font.size": 7})
from matplotlib.figure import Figure

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

jax.config.update("jax_enable_x64", True)

sys.path.append("../src")

from dataclasses import dataclass
from enum import Enum, auto

import activation as activation_module
import network
import normal
import random_matrix
from tqdm import trange


class Activation(Enum):
    """Enumeration for activation function types."""

    PROBIT = auto()
    PROBIT_RESIDUAL = auto()
    SINE = auto()
    SINE_RESIDUAL = auto()


class Topology(Enum):
    """Enumeration for neural network topology types."""

    SMALL = auto()
    WIDE = auto()
    DEEP = auto()


class Weights(Enum):
    """Enumeration for weight initialization states."""

    INITIALIZED = auto()
    TRAINED = auto()


class Variance(Enum):
    """Enumeration for variance levels in the neural network."""

    SMALL = auto()
    MEDIUM = auto()
    LARGE = auto()


class Method(Enum):
    """Enumeration for UQ methods."""

    LINEAR = auto()
    UNSCENTED = auto()
    MEAN_FIELD = auto()
    ANALYTIC = auto()


def build_network(key: jax.Array, activation_type: Activation, topology: Topology):
    logger.info(
        f"Building network with topology={topology.name}, activation={activation_type.name}"
    )
    if activation_type in (Activation.PROBIT, Activation.PROBIT_RESIDUAL):
        layer_args = dict(
            A=random_matrix.RandomOrthogonalProjection(),
            b=random_matrix.ZeroMatrix(),
            activation=activation_module.NormalCDF(),
        )
        if activation_type == Activation.PROBIT:
            hidden_factory = network.Layer.create_nonlinear
        elif activation_type == Activation.PROBIT_RESIDUAL:
            hidden_factory = network.Layer.create_residual
    elif activation_type in (Activation.SINE, Activation.SINE_RESIDUAL):
        layer_args = dict(
            A=random_matrix.RandomOrthogonalProjection(),
            b=random_matrix.RandomUniform(),
            activation=activation_module.Sinusoid(),
        )
        if activation_type == Activation.SINE:
            hidden_factory = network.Layer.create_nonlinear
        elif activation_type == Activation.SINE_RESIDUAL:
            hidden_factory = network.Layer.create_residual

    if topology == Topology.SMALL:
        num_hidden_neurons = 50
        num_hidden_layers = 2
    elif topology == Topology.WIDE:
        num_hidden_neurons = 400
        num_hidden_layers = 2
    elif topology == Topology.DEEP:
        num_hidden_neurons = 100
        num_hidden_layers = 8

    keys = jax.random.split(key, num_hidden_layers + 1)
    layers = [
        network.Layer.create_nonlinear(
            in_size=1,
            out_size=num_hidden_neurons,
            key=keys[0],
            **layer_args,
        )
    ]
    for i in range(1, num_hidden_layers):
        layers.append(
            hidden_factory(
                in_size=num_hidden_neurons,
                out_size=num_hidden_neurons,
                key=keys[i],
                **layer_args,
            )
        )
    layers.append(
        network.Layer.create_linear(
            in_size=num_hidden_neurons,
            out_size=1,
            key=keys[-1],
            C=random_matrix.RandomOrthogonalProjection(),
            d=random_matrix.ZeroMatrix(),
        )
    )
    return network.Network(*layers)


@dataclass
class RandomNeuralNetwork:
    """Test case for random neural network configurations.

    Attributes:
        topology: The network topology type (small, wide, deep)
        weights: The weight initialization state (initialized, trained)
        activation: The activation function type (probit, probit_residual, sine, sine_residual)
    """

    topology: Topology
    weights: Weights
    activation: Activation

    def __post_init__(self):
        logger.info(
            f"Initializing test case: topology={self.topology.name}, "
            f"weights={self.weights.name}, activation={self.activation.name}"
        )

        self.network = build_network(
            jax.random.PRNGKey(1), self.activation, self.topology
        )
        if self.weights == Weights.TRAINED:
            logger.info("Training network...")
            self.train_x, self.train_y = jax.random.normal(
                jax.random.PRNGKey(-1), (2, 10)
            )
            self.network = self.train_network()
            logger.info("Network training completed")

    def train_network(self, learning_rate: float = 1e-4):
        logger.info(f"Starting network training with learning rate={learning_rate}")

        @equinox.filter_jit
        def get_loss(model):
            pred_x = jax.vmap(model)(self.train_x.reshape(-1, 1)).reshape(-1)
            return ((pred_x - self.train_y) ** 2).mean()

        loss_value_and_grad = equinox.filter_value_and_grad(get_loss)

        opt = optax.adamw(learning_rate=learning_rate, weight_decay=1e-3)
        opt_state = opt.init(self.network)

        @equinox.filter_jit
        def step(model, opt_state):
            loss, grads = loss_value_and_grad(model)
            updates, opt_state = opt.update(grads, opt_state, params=model)
            done = loss < 1e-8
            return loss, equinox.apply_updates(model, updates), opt_state, done

        pbar = trange(100000)
        for i in pbar:
            loss, updated_network, opt_state, done = step(self.network, opt_state)
            if done:
                logger.info(
                    f"Training converged after {i} iterations with final loss={loss:.20f}"
                )
                break
            self.network = updated_network
            if i % 100 == 0:
                pbar.set_postfix(
                    {
                        "mse": f"{loss:.20f}",
                    }
                )
        else:
            raise Exception("Training did not converge")

        return self.network

    def plot_function(self):
        logger.info("Plotting function")
        x_grid = np.linspace(-2, 2, 2000)
        y_values = jax.vmap(self.network)(x_grid.reshape(-1, 1)).reshape(-1)

        fig = Figure(dpi=300, figsize=(4, 2), constrained_layout=1)
        ax = fig.gca()
        if (
            self.weights == Weights.TRAINED
            and hasattr(self, "train_x")
            and hasattr(self, "train_y")
        ):
            ax.scatter(
                self.train_x,
                self.train_y,
                color="C1",
                label="training data",
            )
        ax.plot(x_grid, y_values, label="$y = f(x)$")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.legend()
        fig.savefig(f"functions/{str(self)}.pdf")

    def __str__(self) -> str:
        """Return a string representation of the test case."""
        return f"RandomNeuralNetwork(topology={self.topology.name.lower()},weights={self.weights.name.lower()},activation={self.activation.name.lower()})"


@dataclass
class RandomNeuralNetworkTestCase:
    network: RandomNeuralNetwork
    variance: Variance
    num_samples: int = 2**15

    def __post_init__(self):
        self.dist = {
            Variance.SMALL: normal.Normal(jnp.array([0.0]), jnp.array([[1e-4]])),
            Variance.MEDIUM: normal.Normal(jnp.array([0.0]), jnp.array([[1e0]])),
            Variance.LARGE: normal.Normal(
                jnp.array([0.0]),
                jnp.array([[1e4]]),
            ),
        }[self.variance]

        logger.info("Generating quasi-Monte Carlo samples")
        self.monte_carlo_inputs = self.dist.qmc(self.num_samples)
        self.monte_carlo_outputs = jax.vmap(self.network.network)(
            self.monte_carlo_inputs.reshape(-1, 1)
        ).reshape(-1)

        logger.info("Computing normal distributions")
        self.pseudo = normal.Normal.from_samples(self.monte_carlo_outputs)
        self.approximations = {
            Method.LINEAR: self.network.network(self.dist, method="linear"),
            Method.UNSCENTED: self.network.network(self.dist, method="unscented"),
            Method.MEAN_FIELD: self.network.network(
                self.dist, method="analytic", mean_field=True
            ),
            Method.ANALYTIC: self.network.network(
                self.dist, method="analytic", mean_field=False
            ),
        }

    def write_table(self):
        DISTRIBUTION = "distribution"
        MEAN = r"\(\mu\)"
        VARIANCE = r"\(\sigma^2\)"
        WASSERSTEIN = r"\(d_{\mathrm W}(\cdot, Y_0)\)"
        KL = r"\(D_{\mathrm{KL}}(\cdot, Y_1)\)"
        df = pd.DataFrame(
            [
                {
                    DISTRIBUTION: r"pseudo-true (\(Y_1\))",
                    MEAN: self.pseudo.μ.item(),
                    VARIANCE: self.pseudo.Σ.item(),
                    WASSERSTEIN: scipy.stats.wasserstein_distance(
                        self.monte_carlo_outputs.reshape(-1),
                        self.pseudo.qmc(self.num_samples).reshape(-1),
                    ).item(),
                    KL: 0,
                },
                *[
                    {
                        DISTRIBUTION: name,
                        MEAN: dist.μ.item(),
                        VARIANCE: dist.Σ.item(),
                        WASSERSTEIN: scipy.stats.wasserstein_distance(
                            self.monte_carlo_outputs.reshape(-1),
                            dist.qmc(self.num_samples).reshape(-1),
                        ).item(),
                        KL: self.pseudo.kl_divergence(dist).item(),
                    }
                    for name, dist in [
                        (
                            r"\midrule {\bfseries analytic approximation (\(Y\))}",
                            self.approximations[Method.ANALYTIC],
                        ),
                        (
                            r"mean-field approximation",
                            self.approximations[Method.MEAN_FIELD],
                        ),
                        (r"linear approximation", self.approximations[Method.LINEAR]),
                        (
                            r"unscented approximation",
                            self.approximations[Method.UNSCENTED],
                        ),
                    ]
                ]
            ]
        )

        df.to_latex(
            f"tables/{str(self)}.tex",
            index=False,
            escape=False,
            float_format=lambda x: f"{x:.6e}",
            column_format="crrrr",
        )

    def plot_distributions(self):
        y_mesh = np.linspace(
            self.pseudo.μ - 3 * self.pseudo.Σ**0.5,
            self.pseudo.μ + 3 * self.pseudo.Σ**0.5,
            3000,
        ).reshape(-1)
        fig = Figure(dpi=300, figsize=(5, 3), constrained_layout=1)
        ax1 = fig.add_subplot(211)
        ax1.hist(
            self.monte_carlo_outputs, bins=100, density=True, alpha=0.5, label="$Y_0$"
        )
        ax1.plot(
            y_mesh,
            jax.vmap(self.pseudo.pdf)(y_mesh),
            label="$Y_1$",
        )
        ax1.plot(
            y_mesh,
            jax.vmap(self.approximations[Method.ANALYTIC].pdf)(y_mesh),
            label="$Y$",
            linestyle="--",
        )
        ax1.set_xticks([])
        ax1.legend()

        ax2 = fig.add_subplot(212)
        ax2.plot(
            y_mesh,
            jax.vmap(self.approximations[Method.UNSCENTED].pdf)(y_mesh),
            label="unscented",
            linestyle="-",
        )

        ax2.plot(
            y_mesh,
            jax.vmap(self.approximations[Method.LINEAR].pdf)(y_mesh),
            label="linear",
            linestyle="--",
        )

        ax2.plot(
            y_mesh,
            jax.vmap(self.approximations[Method.MEAN_FIELD].pdf)(y_mesh),
            label="mean-field",
            linestyle="dotted",
        )

        ax2.set_ylim(ax1.get_ylim())
        ax2.legend()

        ax2.set_xlabel("$y$")

        fig.savefig(f"distributions/{str(self)}.pdf")

    def __str__(self) -> str:
        return f"RandomNeuralNetworkTestCase(network={self.network},variance={self.variance})"


def generate_networks():
    for topology in Topology:
        for weights in Weights:
            for activation in Activation:
                logger.info(
                    f"Generating network: topology={topology.name}, weights={weights.name}, activation={activation.name}"
                )
                yield RandomNeuralNetwork(topology, weights, activation)


if __name__ == "__main__":
    for random_network in generate_networks():
        logger.info(f"Network: {random_network}")
        random_network.plot_function()
        for variance in Variance:
            test_case = RandomNeuralNetworkTestCase(random_network, variance)
            logger.info(f"Test case: {test_case}")
            test_case.write_table()
            test_case.plot_distributions()
        # break
    # break


# f = build_network(jax.random.PRNGKey(1), Activation.SINE_RESIDUAL, Topology.WIDE)
# f = build_network(jax.random.PRNGKey(1), Activation.PROBIT, Topology.DEEP)

from IPython import embed

embed(colors="neutral")
