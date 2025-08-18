import logging
import os
import sys
from enum import Enum
from typing import Union

import equinox
import jax
import matplotlib
import numpy as np
import optax
import sklearn.metrics
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
import random_matrix
from normal import Normal
from tqdm import tqdm, trange
from unscented import UnscentedTransformMethod

base_path = "../docs/manuscript/generated/"
FAST = False

import ucimlrepo


def load_data():
    taiwanese_bankruptcy_prediction = ucimlrepo.fetch_ucirepo(id=572)

    X = taiwanese_bankruptcy_prediction.data.features.loc[
        :,
        (
            taiwanese_bankruptcy_prediction.data.features
            != taiwanese_bankruptcy_prediction.data.features.iloc[0]
        ).any(),
    ]
    test_features = np.where([("Operating Gross Margin" in c) for c in X.columns])[0]
    logger.info(f"Test features: {X.columns[test_features]}")

    X = X.to_numpy(dtype="float64")
    y = taiwanese_bankruptcy_prediction.data.targets.to_numpy(dtype="int")

    permutation = jax.random.permutation(jax.random.PRNGKey(1), X.shape[0])
    X = X[permutation]
    y = y[permutation]

    train_size = int(len(X) * 0.7)
    test_size = int(len(X) * 0.2)

    mean_x = X[:train_size].mean(axis=0)
    std_x = X[:train_size].std(axis=0)
    X = (X - mean_x) / std_x

    train_x, test_x, val_x = (
        X[:train_size],
        X[train_size : train_size + test_size],
        X[train_size + test_size :],
    )
    train_y, test_y, val_y = (
        y[:train_size],
        y[train_size : train_size + test_size],
        y[train_size + test_size :],
    )

    return (
        train_x,
        train_y,
        val_x,
        val_y,
        test_x,
        test_y,
        test_features,
    )

    # imputation
    # population = Normal.from_samples(train_x)

    # def impute(x_nullable):
    #     x_nullable = (x_nullable - mean_x) / std_x
    #     null_mask = np.isnan(x_nullable)
    #     # projection onto the missing data
    #     P_null = np.eye(x_nullable.shape[0])[null_mask]
    #     P_nonnull = np.eye(x_nullable.shape[0])[~null_mask]

    #     covariance_missing = P_null @ population.Σ @ P_null.T
    #     covariance_nonmissing = P_nonnull @ population.Σ @ P_nonnull.T
    #     covariance_cross = P_null @ population.Σ @ P_nonnull.T

    #     pred_covariance = covariance_missing - covariance_cross @ np.linalg.solve(
    #         covariance_nonmissing,
    #         covariance_cross.T,
    #     )
    #     pred_mean = P_null @ population.μ + covariance_cross @ np.linalg.solve(
    #         covariance_nonmissing,
    #         x_nullable[~null_mask] - population.μ[~null_mask],
    #     )

    #     imputed = np.array(x_nullable)
    #     imputed[null_mask] = pred_mean

    #     # this is a hack
    #     P = np.eye(x_nullable.shape[0])[null_mask]
    #     return Normal(jnp.array(imputed), jnp.array(P.T @ pred_covariance @ P))

    # logger.info("Imputing test data")
    # X_nullable = X[test_size:].to_numpy()
    # imputed_test_x = [impute(x) for x in tqdm(X_nullable)]
    # test_y = y[test_size:].to_numpy(dtype="float64")
    # return train_x, train_y, val_x, val_y, imputed_test_x, test_y


(
    train_x,
    train_y,
    val_x,
    val_y,
    test_x,
    test_y,
    test_features,
) = load_data()


def initialize_network(
    in_size=train_x.shape[1], out_size=1, key=jax.random.PRNGKey(12)
):
    hidden_size = 100

    return network.Network(
        network.Layer.create_nonlinear(
            in_size=in_size,
            out_size=hidden_size,
            key=jax.random.fold_in(key, 0),
            activation=activation.Sinusoid(),
            A=random_matrix.RandomGaussian(),
            b=random_matrix.RandomUniform(),
        ),
        network.Layer.create_linear(
            in_size=hidden_size,
            out_size=out_size,
            key=jax.random.fold_in(key, 1),
            C=random_matrix.ZeroMatrix(),
            d=random_matrix.ConstantMatrix(
                jax.scipy.stats.norm.ppf(train_y.mean(axis=0).item())
            ),
        ),
    )


@equinox.filter_jit
def get_log_probabilities(probit_score: Union[Normal, jnp.ndarray]):
    """Given a probit score, return the log probabilities of the two classes: P(y=0) and P(y=1)."""
    if isinstance(probit_score, Normal):
        μ = probit_score.μ.reshape(-1)
        Σ = probit_score.Σ.reshape(-1)
        assert len(μ) == len(Σ) == 1
        ξ = μ * (1 + Σ) ** (-0.5)
        return jax.scipy.stats.norm.logsf(ξ), jax.scipy.stats.norm.logcdf(ξ)
    else:
        return jax.scipy.stats.norm.logsf(probit_score), jax.scipy.stats.norm.logcdf(
            probit_score
        )


@equinox.filter_jit
def get_loss_single(model, x, y):
    log_p0, log_p1 = get_log_probabilities(model(x))
    return -y * log_p1 - (1 - y) * log_p0


@equinox.filter_jit
def get_loss(model, x, y):
    return jax.vmap(get_loss_single, in_axes=(None, 0, 0))(model, x, y).mean()


def get_network():

    with jax.debug_key_reuse(True):
        f = initialize_network()

    if os.path.exists("assets/bankruptcy.eqx"):
        logger.info("Loading network from file")
        try:
            with open("assets/bankruptcy.eqx", "rb") as serialized:
                f = equinox.tree_deserialise_leaves(serialized, f)
        except:
            logger.info(
                "Failed to load network from file. Maybe the architecture is different?"
            )
            raise
    else:
        logger.info("Training network from scratch")

        loss_value_and_grad = equinox.filter_value_and_grad(get_loss)

        opt = optax.adamw(learning_rate=1e-5)
        opt_state = opt.init(f)

        @equinox.filter_jit
        def step(model, opt_state, x, y):
            loss, grads = loss_value_and_grad(model, x, y)
            updates, opt_state = opt.update(grads, opt_state, params=model)
            return loss, equinox.apply_updates(model, updates), opt_state

        loss_history = []
        val_loss_history = []

        with jax.default_device(jax.devices("cpu")[0]):
            pbar = trange(5000)
            try:
                for i in pbar:
                    loss, f, opt_state = step(f, opt_state, train_x, train_y)

                    decimation = 5
                    if i % decimation == 0:
                        val_loss = get_loss(f, val_x, val_y)
                        loss_history.append(loss)
                        val_loss_history.append(val_loss)
                        pbar.set_postfix(
                            {
                                "loss": f"{loss:.20f}",
                                "val_loss": f"{val_loss:.20f}",
                            }
                        )
                    if i % 500 == 0:
                        fig = Figure(dpi=300, figsize=(4, 2), constrained_layout=1)
                        ax = fig.gca()
                        ax.plot(loss_history[1:])
                        val_loss_history_numpy = np.array(val_loss_history)
                        val_loss_colors = [
                            "C1" if d > 0 else "C2"
                            for d in val_loss_history_numpy[1:]
                            - val_loss_history_numpy[:-1]
                        ]
                        for i in range(1, len(val_loss_history_numpy)):
                            ax.plot(
                                [i - 1, i],
                                [
                                    val_loss_history_numpy[i - 1],
                                    val_loss_history_numpy[i],
                                ],
                                color=val_loss_colors[i - 1],
                            )

                        # ax.legend()
                        ax.set_xlabel(f"epochs / {decimation}")
                        ax.set_ylabel("negative log-likelihood")
                        ax.set_title("Training Losses")
                        fig.savefig("/tmp/losses.pdf")
            except KeyboardInterrupt:
                pass

            with open("assets/bankruptcy.eqx", "wb") as serialized:
                equinox.tree_serialise_leaves(serialized, f)

    return f


f = get_network()

# logger.info("Computing ROC of train")
# log_probabilities_train = np.array([get_log_probabilities(f(z)) for z in tqdm(train_x)])
# logger.info("Computing ROC of val")
# log_probabilities_val = np.array([get_log_probabilities(f(z)) for z in tqdm(val_x)])

# fpr_train, tpr_train, _ = sklearn.metrics.roc_curve(
#     train_y, log_probabilities_train[:, 1]
# )
# fpr_val, tpr_val, _ = sklearn.metrics.roc_curve(val_y, log_probabilities_val[:, 1])

# roc_auc_train = sklearn.metrics.auc(fpr_train, tpr_train)
# roc_auc_val = sklearn.metrics.auc(fpr_val, tpr_val)

# fig = Figure(figsize=(4, 3), constrained_layout=True)
# ax = fig.add_subplot()
# ax.plot(fpr_train, tpr_train, color="C1", label=f"Train (AUC={roc_auc_train:.3f})")
# ax.plot(fpr_val, tpr_val, color="C2", label=f"Val (AUC={roc_auc_val:.3f})")
# ax.plot([0, 1], [0, 1], "k--")
# ax.set_xlabel("False Positive Rate")
# ax.set_ylabel("True Positive Rate")
# ax.set_title("Receiver Operating Characteristic")
# ax.legend()
# fig.savefig("../docs/manuscript/generated/classification/roc-train.pdf")


population = Normal.from_samples(train_x)
P = np.eye(val_x.shape[1])[test_features, :]


def censor_and_impute(x):
    return population.condition_on_projection(P, P @ x)


imputed = [censor_and_impute(z) for z in test_x]
certain = [z.μ for z in imputed]


class UQMethod(Enum):
    CERTAIN = "certain"
    ANALYTIC = "analytic"
    MEAN_FIELD = "mean field"
    LINEAR = "linear"
    UNSCENTED_95 = "unscented'95"
    UNSCENTED_02 = "unscented'02"


@equinox.filter_jit
def get_uq_log_probabilities(z: Normal, method: UQMethod):
    probability_network = network.Network(
        *f.layers,
        network.Layer.create_nonlinear(
            in_size=1,
            out_size=1,
            activation=activation.NormalCDF(offset=0, scale=1),
            A=jnp.eye(1),
            b=jnp.zeros(1),
        ),
    )
    if method == UQMethod.CERTAIN:
        p = probability_network(z.μ)
    elif method == UQMethod.ANALYTIC:
        p = probability_network(z, method="analytic", mean_field=False).μ
    elif method == UQMethod.MEAN_FIELD:
        p = probability_network(z, method="analytic", mean_field=True).μ
    elif method == UQMethod.LINEAR:
        p = probability_network(z, method="linear").μ
    elif method == UQMethod.UNSCENTED_95:
        p = probability_network(
            z, method="unscented", unscented_method=UnscentedTransformMethod.UT0_VECTOR
        ).μ
    elif method == UQMethod.UNSCENTED_02:
        p = probability_network(
            z, method="unscented", unscented_method=UnscentedTransformMethod.UT1_VECTOR
        ).μ
    else:
        raise ValueError(f"Invalid UQ method: {method}")
    return [jnp.log(1 - p), jnp.log(p)]


log_probabilities_full = np.array([get_log_probabilities(f(z)) for z in tqdm(test_x)])
log_probabilities_imputed = {
    method: np.array([get_uq_log_probabilities(z, method) for z in tqdm(imputed)])
    for method in UQMethod
}
log_probabilities_certain = np.array(
    [get_log_probabilities(f(z)) for z in tqdm(certain)]
)

logger.info("Computing ROC of test")
fpr_full, tpr_full, _ = sklearn.metrics.roc_curve(test_y, log_probabilities_full[:, 1])
roc_auc_full = sklearn.metrics.auc(fpr_full, tpr_full)
fpr_certain, tpr_certain, _ = sklearn.metrics.roc_curve(
    test_y, log_probabilities_certain[:, 1]
)
roc_auc_certain = sklearn.metrics.auc(fpr_certain, tpr_certain)
fig = Figure(figsize=(4, 3), constrained_layout=True)
ax = fig.add_subplot()
ax.plot(fpr_full, tpr_full, label=f"Full (AUC={roc_auc_full:.3f})", linestyle="-.")
# ax.plot(fpr_certain, tpr_certain, label=f"Certain (AUC={roc_auc_certain:.3f})")
for method in UQMethod:
    fpr_imputed, tpr_imputed, _ = sklearn.metrics.roc_curve(
        test_y, log_probabilities_imputed[method][:, 1]
    )
    roc_auc_imputed = sklearn.metrics.auc(fpr_imputed, tpr_imputed)

    ax.plot(
        fpr_imputed,
        tpr_imputed,
        label=f"{method.value} (AUC={roc_auc_imputed:.3f})",
    )
ax.plot([0, 1], [0, 1], "k--")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Receiver Operating Characteristic")
ax.legend()
fig.savefig("../docs/manuscript/generated/classification/roc-test.pdf")

imputed_normals = jax.vmap(censor_and_impute)(test_x)

for method in UQMethod:
    log_p0 = log_probabilities_imputed[method][:, 0]
    log_p1 = log_probabilities_imputed[method][:, 1]
    loss = test_y * log_p1 + (1 - test_y) * log_p0
    print(
        f"{method.value} & {loss.mean():.3f} \\ensuremath{{\\pm}} \\num{{{loss.std() / np.sqrt(len(loss)):.1e}}}"
    )
    # loss = (test_y * np.exp(log_p1) + (1 - test_y) * np.exp(log_p0)).mean()
    # print(f"{method.value} prob: {loss:.3f}")
print(f"Full log-prob: {-get_loss(f, test_x, test_y):.3f}")
