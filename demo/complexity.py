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

from enum import Enum, auto

import activation
import network
import random_matrix
from normal import Normal
from unscented import UnscentedTransformMethod


def build_network(key: jax.Array = jax.random.PRNGKey(0)):
    num_hidden_neurons = 400
    num_hidden_layers = 2

    keys = jax.random.split(key, num_hidden_layers + 1)
    # first hidden layer
    layers = [
        network.Layer.create_nonlinear(
            in_size=1,
            out_size=num_hidden_neurons,
            key=keys[0],
            A=random_matrix.RandomGaussian(),
            b=random_matrix.RandomUniform(),
            activation=activation.Sinusoid(),
        )
    ]
    # rest of the hidden layers
    for i in range(1, num_hidden_layers):
        layers.append(
            network.Layer.create_nonlinear(
                in_size=num_hidden_neurons,
                out_size=num_hidden_neurons,
                key=keys[i],
                A=random_matrix.RandomGaussian(),
                b=random_matrix.RandomUniform(),
                activation=activation.Sinusoid(),
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


class Method(Enum):
    """Enumeration for UQ methods."""

    LINEAR = auto()
    UNSCENTED0 = auto()
    UNSCENTED1 = auto()
    MEAN_FIELD = auto()
    ANALYTIC = auto()


f = build_network()


@equinox.filter_jit
def propagate(x: Normal, method: Method):
    if method == Method.LINEAR:
        return f(x, method="linear")
    elif method == Method.UNSCENTED0:
        return f(
            x, method="unscented", unscented_method=UnscentedTransformMethod.UT0_SCALAR
        )
    elif method == Method.UNSCENTED1:
        return f(
            x, method="unscented", unscented_method=UnscentedTransformMethod.UT1_SCALAR
        )
    elif method == Method.MEAN_FIELD:
        return f(x, method="analytic", mean_field=True)
    elif method == Method.ANALYTIC:
        return f(x, method="analytic", mean_field=False)


import functools

# import IPython

# IPython.embed(colors="neutral")
cost = jax.jit(lambda x: f(x)).trace(jnp.array([0.0])).lower().compile().cost_analysis()
print(
    "no uncertainty",
    "&",
    int(cost["transcendentals"]),
    "&",
    int(cost["flops"]),
    "&",
    int(cost["bytes accessed"]),
    "\\\\",
)
for method in Method:
    cost = (
        jax.jit(functools.partial(propagate, method=method))
        .trace(Normal.standard(1))
        .lower()
        .compile()
        .cost_analysis()
    )
    print(
        method.name,
        "&",
        int(cost["transcendentals"]),
        "&",
        int(cost["flops"]),
        "&",
        int(cost["bytes accessed"]),
        "\\\\",
    )
import IPython

IPython.embed(colors="neutral")
