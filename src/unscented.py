from enum import Enum, auto

import equinox
import jax
import numpy as np
from jax import numpy as jnp


class UnscentedHyperparameters(equinox.Module):
    alpha: float
    beta: float
    kappa: float


class UnscentedTransformMethod(Enum):
    """
    Possible methods used to compute the weights and points for the unscented transform.
    """

    # A New Method for the Nonlinear Transformation of Means and Covariances in Filters and Estimators
    UT0_SCALAR = auto(UnscentedHyperparameters(alpha=1, beta=0, kappa=2))
    # The Scaled Unscented Transformation
    UT1_SCALAR = auto(UnscentedHyperparameters(alpha=1e-3, beta=2, kappa=2))


@equinox.filter_jit
def unscented_transform(
    f,
    mu,
    Sigma,
    hyperparameters: UnscentedTransformMethod = UnscentedTransformMethod.UT0_SCALAR,
    verbose=0,
):
    """
    Unscented Transformation for x ~ N(mu, Sigma), where f is a nonlinear function.

    Parameters:
        f     : function from R^n to R^m
        mu    : mean vector of x (shape: [n])
        Sigma : covariance matrix of x (shape: [n, n])
        alpha, beta, kappa: UT tuning parameters
            - alpha: controls how much the sigma points are spread out.
            - beta: used to incorporate prior knowledge of the distribution
            - kappa: secondary scaling parameter

    Returns:
        y_mean: mean of transformed variable
        y_cov : covariance of transformed variable
    """
    hyperparameters = hyperparameters.value
    # with jax.ensure_compile_time_eval():
    n = mu.shape[0]
    lambda_ = hyperparameters.alpha**2 * (n + hyperparameters.kappa) - n

    # alpha controls how much the sigma points are spread out.
    # If alpha is large, the points are more spread out, and the
    # approximation is more accurate for large uncertainties.
    # If alpha is small, the points are closer together, and the
    # approximation is more accurate for small uncertainties.
    # The default value of alpha is set to 1e-3, which is a reasonable
    # value for most applications.

    # Weights for mean and covariance
    Wm = np.full(2 * n + 1, 1 / (2 * (n + lambda_)))
    Wc = np.full(2 * n + 1, 1 / (2 * (n + lambda_)))
    Wm[0] = lambda_ / (n + lambda_)
    Wc[0] = lambda_ / (n + lambda_) + (
        1 - hyperparameters.alpha**2 + hyperparameters.beta
    )

    if verbose > 0:
        print("Wm:", Wm)
        print("Wc:", Wc)

    # Cholesky decomposition of (n + lambda) * Sigma
    sqrt_matrix = jnp.linalg.cholesky((n + lambda_) * (Sigma + 1e-8 * jnp.eye(n)))

    # Generate sigma points
    sigma_points = [mu]
    for i in range(n):
        sigma_points.append(mu + sqrt_matrix[:, i])
        sigma_points.append(mu - sqrt_matrix[:, i])
    sigma_points = jnp.array(sigma_points)  # shape: [2n+1, n]

    if verbose > 0:
        print("sigma_points:", sigma_points)

    # Propagate through the nonlinear function
    Y = jax.vmap(f)(sigma_points)  # shape: [2n+1, m]

    # Compute weighted mean
    y_mean = jnp.sum(Wm[:, None] * Y, axis=0)

    # Compute weighted covariance
    y_cov = jnp.zeros((Y.shape[1], Y.shape[1]))
    for i in range(2 * n + 1):
        diff = Y[i] - y_mean
        y_cov = y_cov + Wc[i] * jnp.outer(diff, diff)

    return y_mean, y_cov
