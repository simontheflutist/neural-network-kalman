import abc

import equinox
import jax
import numpy as np
from jax import numpy as jnp
from jax.scipy.stats.norm import cdf as Φ
from jax.scipy.stats.norm import pdf as ϕ


def ϕ_2(h, k, rho):
    """bivariate normal pdf"""
    return (
        1
        / (2 * np.pi)
        / jnp.sqrt((1 - rho**2))
        * jnp.exp(-0.5 / (1 - rho**2) * (h**2 + k**2 - 2 * rho * h * k))
    )


class Activation(equinox.Module):
    @abc.abstractmethod
    def __call__(self, x):
        raise NotImplementedError

    @abc.abstractmethod
    def M(self, mean, var):
        raise NotImplementedError

    @abc.abstractmethod
    def K(self, mean_1, mean_2, var_1, var_2, covariance):
        raise NotImplementedError

    @abc.abstractmethod
    def L(self, mean_1, mean_2, var_1, var_2, covariance):
        raise NotImplementedError


class Zero(Activation):
    def __call__(self, x):
        return 0

    def M(self, mean, var):
        return 0

    def K(self, mean_1, mean_2, var_1, var_2, covariance):
        return 0

    def L(self, mean_1, mean_2, var_1, var_2, covariance):
        return 0


class ReLU(Activation):
    def __call__(self, x):
        return jnp.maximum(0, x)

    def M(self, mean, var):
        std = var**0.5
        return mean * Φ(mean / std) + std * ϕ(mean / std)

    def K(self, mean_1, mean_2, var_1, var_2, covariance):
        std_1 = var_1**0.5
        std_2 = var_2**0.5
        correlation = jnp.where(covariance != 0, covariance / (std_1 * std_2), 0.0)

        cross_term_1 = mean_1 * (mean_2 * Φ(mean_2 / std_2) + std_2 * ϕ(mean_2 / std_2))
        cross_term_2 = mean_2 * (mean_1 * Φ(mean_1 / std_1) + std_1 * ϕ(mean_1 / std_1))
        return (
            std_1
            * std_2
            * (
                correlation
                * NormalCDF.Φ_2_increment_quad(
                    mean_1 / std_1, mean_2 / std_2, correlation, num_points=20
                )
                + ϕ_2(mean_1 / std_1, mean_2 / std_2, correlation)
            )
            + cross_term_1
            + cross_term_2
            - mean_1 * mean_2
        )

    def L(self, mean_1, mean_2, var_1, var_2, covariance):
        return covariance * Φ(mean_1 * var_1**-0.5)


class Sinusoid(Activation):
    def __call__(self, x):
        return jnp.sin(x)

    def M(self, mean, var):
        return jnp.exp(-var / 2) * jnp.sin(mean)

    def K(self, mean_1, mean_2, var_1, var_2, covariance):
        var_ = -(var_1 + var_2) / 2
        term_1 = (
            0.5
            * (jnp.exp(var_ + covariance) - jnp.exp(var_))
            * jnp.cos(mean_1 - mean_2)
        )
        term_2 = (
            0.5
            * (jnp.exp(var_ - covariance) - jnp.exp(var_))
            * jnp.cos(mean_1 + mean_2)
        )
        return term_1 - term_2

    def L(self, mean_1, mean_2, var_1, var_2, covariance):
        return covariance * jnp.exp(-var_1 / 2) * jnp.cos(mean_1)


class NormalCDF(Activation):
    def __call__(self, x):
        return 2 * Φ(x) - 1

    def M(self, mean, var):
        return self.__call__(mean / (1 + var) ** 0.5)

    def K(self, mean_1, mean_2, var_1, var_2, covariance):
        return 4 * NormalCDF.Φ_2_increment_quad(
            mean_1 / (1 + var_1) ** 0.5,
            mean_2 / (1 + var_2) ** 0.5,
            covariance / ((1 + var_1) ** 0.5 * (1 + var_2) ** 0.5),
        )

    def L(self, mean_1, mean_2, var_1, var_2, covariance):
        return 2 * covariance * (1 + var_1) ** (-0.5) * ϕ(mean_1 / (1 + var_1) ** 0.5)

    @staticmethod
    def gauss_legendre_on_0_x(n, x):
        """
        Generate Gauss-Legendre quadrature nodes and weights for the interval [0, x].

        Parameters:
        - n: int, number of quadrature points
        - x: float, right endpoint of the interval [0, x]

        Returns:
        - nodes: ndarray, transformed quadrature nodes in [0, x]
        - weights: ndarray, transformed quadrature weights
        """
        # Get nodes and weights for [-1, 1]
        nodes, weights = np.polynomial.legendre.leggauss(n)

        # Affine transformation from [-1, 1] to [0, x]
        nodes_transformed = 0.5 * (nodes + 1) * x
        weights_transformed = 0.5 * x * weights

        return nodes_transformed, weights_transformed

    @staticmethod
    def dΦ_2__dρ(ρ, h, k):
        numerator = jnp.exp(-(h**2 - 2 * ρ * h * k + k**2) / (2 * (1 - ρ**2)))
        denominator = 2 * jnp.pi * jnp.sqrt(1 - ρ**2)
        return numerator / denominator

    @staticmethod
    @equinox.filter_jit
    def Φ_2_increment_quad(h, k, ρ, num_points=10):
        """https://www.tandfonline.com/doi/epdf/10.1080/00949659008811236"""
        nodes, weights = NormalCDF.gauss_legendre_on_0_x(num_points, ρ)
        integrand_values = jax.vmap(NormalCDF.dΦ_2__dρ, in_axes=(0, None, None))(
            nodes, h, k
        )
        return weights @ integrand_values
