import logging
import sys

import jax
import matplotlib
import numpy as np
import scipy
from jax import numpy as jnp
from matplotlib.figure import Figure

jax.config.update("jax_enable_x64", True)

matplotlib.rcParams.update({"font.size": 7})

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

sys.path.append("../src")

import activation
import network
import normal
import random_matrix


def check_activation(network):
    for layer in network.layers:
        if not (
            isinstance(layer.activation, activation.NormalCDF)
            or isinstance(layer.activation, activation.Zero)
        ):
            raise ValueError("All layers in the network must use NormalCDF activation")


def sample_stochastic_network(
    key: jax.random.PRNGKey,
    network: network.Network,
    x: jnp.ndarray,
    num_samples: int = 1,
) -> jnp.ndarray:
    """
    Sample from a stochastic version of a neural network with NormalCDF activation.

    The stochastic activation is defined as:
        σ̃(x, U) = 2 * 1_{U < Φ(x)} - 1
    where U ~ Uniform(0,1) and Φ is the standard normal CDF.

    Args:
        key: JAX random key for sampling
        network: The neural network with NormalCDF activation
        x: Input to the network (batch_size, input_dim)
        num_samples: Number of samples to generate

    Returns:
        Array of shape (num_samples, batch_size, output_dim) containing the samples
    """
    # Check if the network uses NormalCDF activation
    check_activation(network)

    # Process each sample
    def process_sample(sample_key):
        # Make a copy of the input
        current_x = x

        # Process through each layer
        for i, layer in enumerate(network.layers):
            # Linear transformation: A @ x + b
            current_x = (
                2
                * jax.random.bernoulli(
                    jax.random.fold_in(sample_key, i),
                    jax.scipy.stats.norm.cdf(layer.A @ current_x + layer.b),
                )
                - 1
                + layer.C @ current_x
                + layer.d
            )

        return current_x

    # Generate multiple samples if needed
    keys = jax.random.split(key, num_samples)
    samples = jax.vmap(process_sample)(keys)

    return samples


def stochastic_network_propagate(μ, Σ, network):
    check_activation(network)

    for layer in network.layers:
        preactivation_mean = layer.A @ μ + layer.b
        preactivation_var = jnp.diag(layer.A @ Σ @ layer.A.T)

        μ, Σ = layer._propagate_mean(μ, Σ), layer._propagate_cov(μ, Σ)
        Σ += 8 * jnp.diag(
            scipy.special.owens_t(
                preactivation_mean * (1 + preactivation_var) ** -0.5,
                (1 + 2 * preactivation_var) ** -0.5,
            )
        )

    return μ, Σ


def build_network(in_size):
    num_hidden_neurons = 100
    num_hidden_layers = 8
    keys = jax.random.split(jax.random.PRNGKey(-1), num_hidden_layers + 1)
    layers = [
        network.Layer.create_nonlinear(
            in_size=in_size,
            out_size=num_hidden_neurons,
            key=keys[0],
            A=random_matrix.RandomGaussian(),
            b=random_matrix.RandomGaussian(),
            activation=activation.NormalCDF(),
        )
    ]
    for i in range(1, num_hidden_layers):
        layers.append(
            network.Layer.create_nonlinear(
                in_size=num_hidden_neurons,
                out_size=num_hidden_neurons,
                key=keys[i],
                A=random_matrix.RandomGaussian(),
                b=random_matrix.RandomGaussian(),
                activation=activation.NormalCDF(),
            )
        )
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


if __name__ == "__main__":
    in_size = 1
    network = build_network(in_size)

    x = jnp.zeros(in_size)
    samples = sample_stochastic_network(
        jax.random.PRNGKey(0), network, x, num_samples=100000
    ).reshape(-1)
    pseudo = normal.Normal.from_samples(samples)
    analytic = normal.Normal(
        *stochastic_network_propagate(
            jnp.zeros(in_size), jnp.zeros((in_size, in_size)), network
        )
    )

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
    fig = Figure(dpi=300, figsize=(4, 2), constrained_layout=1)
    ax1 = fig.add_subplot()
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
        jax.vmap(analytic.pdf)(y_mesh),
        label="analytic",
        linestyle="--",
        color="C1",
    )
    ax1.set_xlabel("network output")
    ax1.set_ylabel("probability density")
    ax1.legend()
    fig.savefig("../docs/manuscript/generated/stochastic.pdf")

    print("KL divergence", pseudo.kl_divergence(analytic))

    import IPython

    IPython.embed(colors="neutral")
