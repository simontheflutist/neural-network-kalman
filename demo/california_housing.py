import logging
import os
import sys
from enum import Enum

import equinox
import jax
from jax import numpy as jnp

jax.config.update("jax_enable_x64", True)

import matplotlib
import numpy as np
import optax
import pandas as pd
from matplotlib.figure import Figure

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
import random_matrix
from normal import Normal
from tqdm import tqdm, trange
from unscented import UnscentedTransformMethod

base_path = "../docs/manuscript/generated/"
FAST = False

columns = [
    "longitude",
    "latitude",
    "housingMedianAge",
    "totalRooms",
    "totalBedrooms",
    "population",
    "households",
    "medianIncome",
    "medianHouseValue",
]


def load_data():
    dataset = pd.read_csv(
        "data/CaliforniaHousing/cal_housing.data", header=None, names=columns
    )[columns[:]].to_numpy()
    dataset[:, 3:] = np.log(dataset[:, 3:])

    dataset = jax.random.permutation(jax.random.PRNGKey(12), dataset)

    train_size = int(len(dataset) * 0.7)
    test_size = 4096

    dataset = (dataset - dataset[:train_size].mean(axis=0)) / dataset[:train_size].std(
        axis=0
    )

    train_x, train_y = (
        dataset[:train_size, :-1],
        dataset[:train_size, -1],
    )
    test_x, test_y = (
        dataset[train_size : train_size + test_size, :-1],
        dataset[train_size : train_size + test_size, -1],
    )
    val_x, val_y = (
        dataset[train_size + test_size :, :-1],
        dataset[train_size + test_size :, -1],
    )

    return train_x, train_y, test_x, test_y, val_x, val_y


train_x, train_y, test_x, test_y, val_x, val_y = load_data()


def initialize_network(
    in_size=train_x.shape[1], out_size=1, key=jax.random.PRNGKey(12)
):
    hidden_size = 7

    return network.Network(
        network.Layer.create_nonlinear(
            in_size=in_size,
            out_size=hidden_size,
            key=jax.random.fold_in(key, 0),
            activation=activation.Sinusoid(),
            A=random_matrix.RandomGaussian(),
            b=random_matrix.RandomUniform(),
        )._augment_with_identity(),
        network.Layer.create_linear(
            in_size=in_size + hidden_size,
            out_size=out_size,
            key=jax.random.fold_in(key, 1),
            C=random_matrix.ZeroMatrix(),
            d=random_matrix.ConstantMatrix(train_y.mean()),
        ),
    )


def get_network():

    with jax.debug_key_reuse(True):
        f = initialize_network()

    if os.path.exists("assets/california_housing.eqx"):
        logger.info("Loading network from file")
        try:
            with open("assets/california_housing.eqx", "rb") as serialized:
                f = equinox.tree_deserialise_leaves(serialized, f)
        except:
            logger.info(
                "Failed to load network from file. Maybe the architecture is different?"
            )
            raise
    else:
        logger.info("Training network from scratch")

        @equinox.filter_jit
        def get_loss(model, x, y):
            pred_x = jax.vmap(model)(x).reshape(-1)
            return ((pred_x - y) ** 2).mean()

        loss_value_and_grad = equinox.filter_value_and_grad(get_loss)

        opt = optax.adam(
            learning_rate=optax.join_schedules(
                [optax.constant_schedule(1e-1), optax.constant_schedule(1e-2)], [2000]
            )
        )
        opt_state = opt.init(f)

        @equinox.filter_jit
        def step(model, opt_state, x, y):
            loss, grads = loss_value_and_grad(model, x, y)
            updates, opt_state = opt.update(grads, opt_state, params=model)
            return loss, equinox.apply_updates(model, updates), opt_state

        nll_history = []
        val_nll_history = []

        with jax.default_device(jax.devices("cuda")[0]):
            pbar = trange(8000)
            try:
                for i in pbar:
                    loss, f, opt_state = step(f, opt_state, train_x, train_y)

                    nll = 0.5 + np.log(loss)
                    decimation = 10
                    if i % decimation == 0:
                        val_loss = get_loss(f, val_x, val_y)
                        val_nll = 0.5 * val_loss / loss + np.log(loss)
                        nll_history.append(nll)
                        val_nll_history.append(val_nll)
                        pbar.set_postfix(
                            {
                                "nll": f"{nll:.20f}",
                                "val_nll": f"{val_nll:.20f}",
                            }
                        )
                    if i % 500 == 0:
                        fig = Figure(dpi=300, figsize=(4, 2), constrained_layout=1)
                        ax = fig.gca()
                        ax.plot(nll_history[1:])
                        val_nll_history_numpy = np.array(val_nll_history)
                        val_nll_colors = [
                            "C1" if d > 0 else "C2"
                            for d in val_nll_history_numpy[1:]
                            - val_nll_history_numpy[:-1]
                        ]
                        for i in range(1, len(val_nll_history_numpy)):
                            ax.plot(
                                [i - 1, i],
                                [
                                    val_nll_history_numpy[i - 1],
                                    val_nll_history_numpy[i],
                                ],
                                color=val_nll_colors[i - 1],
                            )

                        # ax.legend()
                        ax.set_xlabel(f"epochs / {decimation}")
                        ax.set_ylabel("negative log-likelihood")
                        ax.set_title("Training Losses")
                        fig.savefig("/tmp/losses.pdf")
            except KeyboardInterrupt:
                pass

            with open("assets/california_housing.eqx", "wb") as serialized:
                equinox.tree_serialise_leaves(serialized, f)

    return f


class UQMethod(Enum):
    CERTAIN = "certain"
    ANALYTIC = "analytic"
    MEAN_FIELD = "mean field"
    LINEAR = "linear"
    UNSCENTED_95 = "unscented'95"
    UNSCENTED_02 = "unscented'02"


if __name__ == "__main__":
    f = get_network()
    noisy_features = slice(2, None)
    noise_covariance = jnp.zeros((train_x.shape[1], train_x.shape[1]))
    noise_covariance = noise_covariance.at[noisy_features, noisy_features].set(
        1e1 * np.cov(train_x, rowvar=0)[2:, 2:]
    )
    input_noise_dist = Normal(μ=jnp.zeros(train_x.shape[1]), Σ=noise_covariance)

    @equinox.filter_jit
    def predict(x, uq: UQMethod = UQMethod.CERTAIN):
        with jax.ensure_compile_time_eval():
            residuals = jax.vmap(f)(train_x).reshape(-1) - train_y
            prediction_noise_dist = Normal.from_samples(residuals)
        x = jnp.asarray(x).reshape(-1)
        model_input = Normal.certain(x)
        if uq == UQMethod.CERTAIN:
            model_output = f(model_input)
        elif uq == UQMethod.ANALYTIC:
            model_output = f(
                model_input + input_noise_dist, method="analytic", mean_field=False
            )
        elif uq == UQMethod.MEAN_FIELD:
            model_output = f(
                model_input + input_noise_dist, method="analytic", mean_field=True
            )
        elif uq == UQMethod.LINEAR:
            model_output = f(model_input + input_noise_dist, method="linear")
        elif uq == UQMethod.UNSCENTED_95:
            model_output = f(
                model_input + input_noise_dist,
                method="unscented",
                unscented_method=UnscentedTransformMethod.UT0_VECTOR,
            )
        elif uq == UQMethod.UNSCENTED_02:
            model_output = f(
                model_input + input_noise_dist,
                method="unscented",
                unscented_method=UnscentedTransformMethod.UT1_VECTOR,
            )
        return model_output + prediction_noise_dist

    def inference_loss(x, y, uq_method: UQMethod):
        return -predict(x, uq_method).lpdf(y)

    @equinox.filter_jit
    def evaluate_uq_method_batch(noisy_x, y, uq_method: UQMethod):
        values = jax.vmap(inference_loss, in_axes=(0, 0, None))(noisy_x, y, uq_method)
        mean = values.mean()
        variance = values.var()
        return mean, variance

    def evaluate_uq_method(x, y, num_batches, uq_method: UQMethod):
        input_noise = input_noise_dist.qmc(num_samples=len(x) * num_batches)
        means_and_variances = [
            evaluate_uq_method_batch(
                x + w,
                y,
                uq_method,
            )
            for w in tqdm(np.split(input_noise, num_batches))
        ]
        means, variances = zip(*means_and_variances)
        mean = np.mean(means)
        std = (np.mean(variances) + np.var(means)) ** 0.5
        return mean, std / num_batches

    for meth in list(UQMethod)[::-1]:
        print(
            meth,
            evaluate_uq_method(test_x, test_y, num_batches=512, uq_method=meth),
        )

    import IPython

    IPython.embed(colors="neutral")
