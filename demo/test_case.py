import logging
import sys

import equinox
import jax

jax.config.update("jax_enable_x64", True)

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


sys.path.append("../src")

from dataclasses import dataclass
from enum import Enum, auto

import activation as activation_module
import network
import normal
import random_matrix
from tqdm import tqdm, trange
from unscented import UnscentedTransformMethod

base_path = "../docs/manuscript/generated/"
FAST = False
IN_SIZE = 3


class Activation(Enum):
    """Enumeration for activation function types."""

    PROBIT = auto()
    PROBIT_RESIDUAL = auto()
    SINE = auto()
    SINE_RESIDUAL = auto()


class Architecture(Enum):
    """Enumeration for neural network architecture types."""

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
    UNSCENTED0 = auto()
    UNSCENTED1 = auto()
    MEAN_FIELD = auto()
    ANALYTIC = auto()


def build_network(
    key: jax.Array, activation_type: Activation, architecture: Architecture
):
    logger.info(
        f"Building network with architecture={architecture.name}, activation={activation_type.name}"
    )
    A_factory = random_matrix.RandomGaussian(1.1)
    if activation_type in (Activation.PROBIT, Activation.PROBIT_RESIDUAL):
        layer_args = dict(
            A=A_factory,
            b=random_matrix.RandomGaussian(),
            activation=activation_module.NormalCDF(),
        )
        if activation_type == Activation.PROBIT:
            hidden_factory = network.Layer.create_nonlinear
        elif activation_type == Activation.PROBIT_RESIDUAL:
            hidden_factory = network.Layer.create_residual
    elif activation_type in (Activation.SINE, Activation.SINE_RESIDUAL):
        layer_args = dict(
            A=A_factory,
            b=random_matrix.RandomUniform(),
            activation=activation_module.Sinusoid(),
        )
        if activation_type == Activation.SINE:
            hidden_factory = network.Layer.create_nonlinear
        elif activation_type == Activation.SINE_RESIDUAL:
            hidden_factory = network.Layer.create_residual

    if architecture == Architecture.SMALL:
        num_hidden_neurons = 50
        num_hidden_layers = 2
    elif architecture == Architecture.WIDE:
        num_hidden_neurons = 400
        num_hidden_layers = 2
    elif architecture == Architecture.DEEP:
        num_hidden_neurons = 100
        num_hidden_layers = 8

    keys = jax.random.split(key, num_hidden_layers + 1)
    # first hidden layer
    layers = [
        network.Layer.create_nonlinear(
            in_size=IN_SIZE,
            out_size=num_hidden_neurons,
            key=keys[0],
            A=A_factory,
            b=layer_args["b"],
            activation=layer_args["activation"],
        )
    ]
    # rest of the hidden layers
    for i in range(1, num_hidden_layers):
        layers.append(
            hidden_factory(
                in_size=num_hidden_neurons,
                out_size=num_hidden_neurons,
                key=keys[i],
                **layer_args,
            )
        )
    # make a provisional output layer to get linearized mean and variance
    layers.append(
        network.Layer.create_linear(
            in_size=num_hidden_neurons,
            out_size=1,
            key=keys[-1],
            C=random_matrix.RandomGaussian(),
            d=random_matrix.ZeroMatrix(),
        )
    )
    return network.Network(*layers)


@dataclass
class RandomNeuralNetwork:
    """Test case for random neural network configurations.

    Attributes:
        architecture: The network architecture type (small, wide, deep)
        weights: The weight initialization state (initialized, trained)
        activation: The activation function type (probit, probit_residual, sine, sine_residual)
    """

    architecture: Architecture
    weights: Weights
    activation: Activation

    def __post_init__(self):
        logger.info(
            f"Initializing test case: architecture={self.architecture.name}, "
            f"weights={self.weights.name}, activation={self.activation.name}"
        )

        self.network = build_network(
            jax.random.PRNGKey(1), self.activation, self.architecture
        )
        if self.weights == Weights.TRAINED:
            logger.info("Training network...")
            NUM_TRAINING_SAMPLES = 10
            self.train_x = jax.random.normal(
                jax.random.PRNGKey(-1), (IN_SIZE, NUM_TRAINING_SAMPLES)
            )
            self.train_y = jax.random.normal(
                jax.random.PRNGKey(-2), (NUM_TRAINING_SAMPLES,)
            )
            self.network = self.train_network()
            logger.info("Network training completed")

    def train_network(self, learning_rate: float = 1e-5):
        logger.info(f"Starting network training with learning rate={learning_rate}")

        @equinox.filter_jit
        def get_loss(model):
            pred_x = jax.vmap(model)(self.train_x.reshape(-1, IN_SIZE)).reshape(-1)
            return ((pred_x - self.train_y) ** 2).mean()

        loss_value_and_grad = equinox.filter_value_and_grad(get_loss)

        opt = optax.adamw(learning_rate=learning_rate, weight_decay=1e-4)
        opt_state = opt.init(self.network)

        @equinox.filter_jit
        def step(model, opt_state):
            loss, grads = loss_value_and_grad(model)
            updates, opt_state = opt.update(grads, opt_state, params=model)
            done = loss < 1e-8
            return loss, equinox.apply_updates(model, updates), opt_state, done

        with jax.default_device(jax.devices("cpu")[0]):
            pbar = trange(1_000_000 if not FAST else 1000)
            for i in pbar:
                loss, updated_network, opt_state, done = step(self.network, opt_state)
                if i >= 30000 and done:
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
                if not FAST:
                    raise Exception("Training did not converge")

        return self.network

    def plot_function(self):
        logger.info("Plotting function")
        x_grid = np.zeros((2000, IN_SIZE))
        x_grid[:, 0] = np.linspace(-2, 2, 2000)
        y_values = jax.vmap(self.network)(x_grid).reshape(-1)

        fig = Figure(dpi=300, figsize=(4, 2), constrained_layout=1)
        ax = fig.gca()
        # if (
        #     self.weights == Weights.TRAINED
        #     and hasattr(self, "train_x")
        #     and hasattr(self, "train_y")
        # ):
        #     ax.scatter(
        #         self.train_x,
        #         self.train_y,
        #         color="C1",
        #         label="training data",
        #     )
        ax.plot(x_grid[:, 0], y_values, label="$y = f(x)$")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.legend()
        filename = f"functions/{str(self)}.pdf"
        fig.savefig(base_path + filename)
        return filename

    def __str__(self) -> str:
        """Return a string representation of the test case."""
        return f"{self.architecture.name.lower()}_{self.weights.name.lower()}_{self.activation.name.lower()}"

    @property
    def pretty_name(self) -> str:
        """Return a nicely formatted name for the network."""
        # Convert to title case and replace underscores with spaces
        architecture = self.architecture.name.lower().replace("_", " ")
        weights = self.weights.name.lower().replace("_", " ")
        activation = self.activation.name.lower().replace("_", " ")
        return f"Network(architecture={architecture}, weights={weights}, activation={activation})"


@dataclass
class RandomNeuralNetworkTestCase:
    network: RandomNeuralNetwork
    variance: Variance
    num_samples: int = 2**16
    num_repetitions: int = 20

    def __post_init__(self):
        self.dist = {
            Variance.SMALL: 1e-1 * normal.Normal.standard(IN_SIZE),
            Variance.MEDIUM: normal.Normal.standard(IN_SIZE),
            Variance.LARGE: 1e1 * normal.Normal.standard(IN_SIZE),
        }[self.variance]

        logger.info("Generating quasi-Monte Carlo samples")
        self.monte_carlo_inputs = [
            self.dist.qmc(self.num_samples, seed=i) for i in range(self.num_repetitions)
        ]

        self.monte_carlo_outputs = [
            jax.vmap(self.network.network)(
                self.monte_carlo_inputs[i].reshape(-1, IN_SIZE)
            ).reshape(-1)
            for i in range(self.num_repetitions)
        ]

        logger.info("Computing normal distributions")
        self.pseudo = [
            normal.Normal.from_samples(self.monte_carlo_outputs[i])
            for i in range(self.num_repetitions)
        ]
        with jax.default_device(jax.devices("cpu")[0]):
            self.approximations = {
                Method.LINEAR: self.network.network(self.dist, method="linear"),
                Method.UNSCENTED0: self.network.network(
                    self.dist,
                    method="unscented",
                    unscented_method=UnscentedTransformMethod.UT0_SCALAR,
                ),
                Method.UNSCENTED1: self.network.network(
                    self.dist,
                    method="unscented",
                    unscented_method=UnscentedTransformMethod.UT1_SCALAR,
                ),
                Method.MEAN_FIELD: self.network.network(
                    self.dist, method="analytic", mean_field=True
                ),
                Method.ANALYTIC: self.network.network(
                    self.dist, method="analytic", mean_field=False
                ),
            }

    @property
    def pretty_name(self):
        return f"{self.network.pretty_name}, variance={self.variance.name.lower()}"

    def write_table(self):
        def bootstrap_mean_std(statistic):
            repetitions = np.array(
                [statistic(samples) for samples in self.monte_carlo_outputs]
            )
            return repetitions.mean(), repetitions.std() / np.sqrt(len(repetitions))

        def format_scientific(x, implicit_plus=True):
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

        DISTRIBUTION = "distribution"
        MEAN = r"\(\mu\)"
        VARIANCE = r"\(\sigma^2\)"
        WASSERSTEIN = r"\(d_{\mathrm W}(\cdot, Y_0)\)"
        KL = r"\(D_{\mathrm{KL}}(\cdot \| Y_1)\)"

        def wasserstein(samples, dist):
            sorted_data = np.sort(samples)
            theoretical_quantiles = scipy.stats.norm(
                loc=dist.μ.item(), scale=dist.Σ.item() ** 0.5
            ).ppf((np.arange(self.num_samples) + 0.5) / (self.num_samples))
            return (
                np.abs(theoretical_quantiles - sorted_data).mean()
                * samples.std().item() ** -0.5
            )

        df = pd.DataFrame(
            [
                {
                    DISTRIBUTION: r"pseudo-true (\(Y_1\))",
                    MEAN: format_scientific_uncertainty(
                        *bootstrap_mean_std(lambda samples: samples.mean()),
                        implicit_plus=True,
                    ),
                    VARIANCE: format_scientific_uncertainty(
                        *bootstrap_mean_std(lambda samples: samples.var()),
                        implicit_plus=False,
                    ),
                    WASSERSTEIN: format_scientific_uncertainty(
                        *bootstrap_mean_std(
                            lambda samples: wasserstein(
                                samples, normal.Normal.from_samples(samples)
                            )
                        ),
                        implicit_plus=False,
                    ),
                    KL: 0,
                },
            ]
        )

        for name, dist in [
            (
                r"\midrule analytic",
                self.approximations[Method.ANALYTIC],
            ),
            (
                r"mean-field",
                self.approximations[Method.MEAN_FIELD],
            ),
            (r"linear", self.approximations[Method.LINEAR]),
            (
                r"unscented'95",
                self.approximations[Method.UNSCENTED0],
            ),
            (
                r"unscented'02",
                self.approximations[Method.UNSCENTED1],
            ),
        ]:
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        [
                            {
                                DISTRIBUTION: name,
                                MEAN: format_scientific(
                                    dist.μ.item(), implicit_plus=True
                                ),
                                VARIANCE: format_scientific(
                                    dist.Σ.item(), implicit_plus=False
                                ),
                                WASSERSTEIN: format_scientific_uncertainty(
                                    *bootstrap_mean_std(
                                        lambda samples: wasserstein(samples, dist)
                                    ),
                                    implicit_plus=False,
                                ),
                                KL: format_scientific_uncertainty(
                                    *bootstrap_mean_std(
                                        lambda samples: normal.Normal.from_samples(
                                            samples
                                        )
                                        .kl_divergence(dist)
                                        .item()
                                    ),
                                    implicit_plus=False,
                                ),
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

        moment_filename = f"tables/moments/{str(self)}.tex"
        divergence_filename = f"tables/divergences/{str(self)}.tex"

        df[df.columns[:3]].to_latex(
            base_path + moment_filename,
            index=False,
            escape=False,
            column_format="cllll",
        )
        df[[df.columns[0], *df.columns[3:]]].to_latex(
            base_path + divergence_filename,
            index=False,
            escape=False,
            column_format="cllll",
        )
        return moment_filename, divergence_filename

    def plot_distributions(self):
        samples = np.array(self.monte_carlo_outputs).flatten()
        pseudo = normal.Normal.from_samples(samples)
        y_mesh = np.linspace(
            min(
                pseudo.μ - 3 * pseudo.Σ**0.5,
                np.percentile(samples, 0.5),
            ),
            max(
                pseudo.μ + 3 * pseudo.Σ**0.5,
                np.percentile(samples, 99.5),
            ),
            3000,
        ).reshape(-1)
        fig = Figure(dpi=300, figsize=(5, 3), constrained_layout=1)
        ax1 = fig.add_subplot(311)
        ax1.hist(
            samples,
            bins=50,
            density=True,
            alpha=0.5,
            label="true",
            color="C2",
        )
        ax1.plot(y_mesh, jax.vmap(pseudo.pdf)(y_mesh), label="pseudo", color="C0")
        ax1.plot(
            y_mesh,
            jax.vmap(self.approximations[Method.ANALYTIC].pdf)(y_mesh),
            label="analytic",
            linestyle="--",
            color="C1",
        )
        ax1.set_ylim(
            (
                ax1.get_ylim()[0] - 0.05 * (ax1.get_ylim()[1] - ax1.get_ylim()[0]),
                ax1.get_ylim()[1],
            )
        )
        ax1.set_xlim(y_mesh.min(), y_mesh.max())
        ax1.set_xticks([])
        ax1.legend()

        ax2 = fig.add_subplot(312)

        ax2.plot(
            y_mesh,
            jax.vmap(self.approximations[Method.LINEAR].pdf)(y_mesh),
            label="linear",
            linestyle="-",
        )

        ax2.plot(
            y_mesh,
            jax.vmap(self.approximations[Method.MEAN_FIELD].pdf)(y_mesh),
            label="mean-field",
            linestyle="--",
        )
        ax2.set_xticks([])
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_ylim(ax1.get_ylim())
        ax2.legend()

        ax3 = fig.add_subplot(313)
        ax3.plot(
            y_mesh,
            jax.vmap(self.approximations[Method.UNSCENTED0].pdf)(y_mesh),
            label="unscented'95",
            linestyle="-",
        )
        ax3.plot(
            y_mesh,
            jax.vmap(self.approximations[Method.UNSCENTED1].pdf)(y_mesh),
            label="unscented'02",
            linestyle="--",
        )
        ax3.set_xlim(ax1.get_xlim())
        ax3.set_ylim(ax1.get_ylim())
        ax3.legend()

        ax3.set_xlabel("$y$")

        filename = f"distributions/{str(self)}.pdf"
        fig.savefig(base_path + filename)
        return filename

    def __str__(self) -> str:
        return f"RandomNeuralNetworkTestCase(network={self.network},variance={self.variance})"


def generate_networks():
    for architecture in Architecture:
        for activation in Activation:
            for weights in Weights:
                logger.info(
                    f"Generating network: architecture={architecture.name}, weights={weights.name}, activation={activation.name}"
                )
                yield RandomNeuralNetwork(architecture, weights, activation)


if __name__ == "__main__":
    with open(base_path + "generated.tex", "w") as f:
        for random_network in generate_networks():
            logger.info(f"Network: {random_network}")
            f.write(r"\subsection{" + random_network.pretty_name + "}\n")
            # filename = random_network.plot_function()
            # f.write(r"\begin{figure}[H]\begin{center}" + "\n")
            # f.write(f"\\includegraphics{{generated/{filename}}}\n")
            # f.write(r"\end{center}" + "\n")
            # f.write(
            #     rf"\caption{{Input-output relationship of {random_network.pretty_name}}}"
            #     + "\n"
            # )
            # f.write(r"\end{figure}" + "\n")
            # f.write("\\clearpage\n")

            for variance in Variance:
                test_case = RandomNeuralNetworkTestCase(random_network, variance)
                f.write(
                    f"\\subsubsection*{{{random_network.pretty_name}, Variance: {variance.name}}}\n"
                )
                logger.info(f"Test case: {test_case}")

                moment_filename, divergence_filename = test_case.write_table()
                distribution_name = test_case.plot_distributions()
                f.write(r"\begin{table}[H]\begin{center}")
                f.write(f"\\input{{generated/{moment_filename}}}\n")
                f.write(r"\end{center}" + "\n")
                f.write(
                    rf"\caption{{Comparison of moments for {test_case.pretty_name}}}"
                    + "\n"
                )
                f.write(r"\end{table}")

                f.write(r"\begin{table}[H]\begin{center}")
                f.write(f"\\input{{generated/{divergence_filename}}}\n")
                f.write(r"\end{center}" + "\n")
                f.write(
                    rf"\caption{{Comparison of statistical distances for {test_case.pretty_name}}}"
                    + "\n"
                )
                f.write(r"\end{table}")

                f.write(r"\begin{figure}[H]\begin{center}" + "\n")
                f.write(f"\\includegraphics{{generated/{distribution_name}}}\n")
                f.write(r"\end{center}" + "\n")
                f.write(
                    rf"\caption{{Probability distributions for {test_case.pretty_name}}}"
                    + "\n"
                )
                f.write(r"\end{figure}")

                f.write("\\clearpage\n")
            f.flush()

            # Uncomment to process only one network for testing
            # break


# f = build_network(jax.random.PRNGKey(1), Activation.SINE_RESIDUAL, Topology.WIDE)
# f = build_network(jax.random.PRNGKey(1), Activation.PROBIT, Topology.DEEP)

# from IPython import embed

# embed(colors="neutral")
