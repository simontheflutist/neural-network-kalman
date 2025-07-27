import equinox
import jax
import numpy as np
from jax import numpy as jnp


@equinox.filter_jit
def unscented_transform(f, mu, Sigma, alpha=1e0, beta=2, kappa=0, verbose=0):
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
    # with jax.ensure_compile_time_eval():
    n = mu.shape[0]
    lambda_ = alpha**2 * (n + kappa) - n

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
    Wc[0] = Wm[0] + (1 - alpha**2 + beta)

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
