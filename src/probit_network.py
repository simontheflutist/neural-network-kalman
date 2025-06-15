import typing

import equinox
import jax
import numpy as np
from jax import numpy as jnp
from jax.scipy.stats.norm import cdf as Φ
from jax.scipy.stats.norm import pdf as ϕ

from random_matrix import RandomMatrixFactory, ZeroMatrix
from unscented import unscented_transform


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


def dΦ_2__dρ(ρ, h, k):
    numerator = jnp.exp(-(h**2 - 2 * ρ * h * k + k**2) / (2 * (1 - ρ**2)))
    denominator = 2 * jnp.pi * jnp.sqrt(1 - ρ**2)
    return numerator / denominator


@equinox.filter_jit
def Φ_2_increment_quad(h, k, ρ, num_points=20):
    """https://www.tandfonline.com/doi/epdf/10.1080/00949659008811236"""
    nodes, weights = gauss_legendre_on_0_x(num_points, ρ)
    integrand_values = jax.vmap(dΦ_2__dρ, in_axes=(0, None, None))(nodes, h, k)
    return weights @ integrand_values


def σ(x):
    return 2.0 * Φ(x) - 1.0


def _M(μ, Σ, a, b, c, d):
    return σ((b + a @ μ) * (1 + a @ Σ @ a) ** (-0.5)) + c @ μ + d


def _K(μ, Σ, a_1, b_1, c_1, a_2, b_2, c_2):
    μ_1 = a_1 @ μ + b_1
    μ_2 = a_2 @ μ + b_2
    σ_1 = (1 + a_1 @ Σ @ a_1) ** 0.5
    σ_2 = (1 + a_2 @ Σ @ a_2) ** 0.5
    ρ = (a_1 @ Σ @ a_2) / (σ_1 * σ_2)

    term_Φ_Φ = 4 * Φ_2_increment_quad(μ_1 / σ_1, μ_2 / σ_2, ρ)
    term_a_c = 2 * (a_1 @ Σ @ c_2) / σ_1 * ϕ(μ_1 / σ_1)
    term_c_a = 2 * (a_2 @ Σ @ c_1) / σ_2 * ϕ(μ_2 / σ_2)

    term_c_c = c_1 @ Σ @ c_2

    return term_Φ_Φ + term_a_c + term_c_a + term_c_c


def rectify_eigenvalues(P):
    Λ, V = jnp.linalg.eigh(P, symmetrize_input=1)
    return V @ jnp.diag(jnp.maximum(Λ, 0)) @ V.T


@equinox.filter_jit
def sum(*a):
    return jnp.sum(jnp.array(a))


class ProbitLinear(equinox.Module):
    A: jax.Array
    b: jax.Array
    C: jax.Array
    d: jax.Array
    in_size: int
    out_size: int

    def __init__(
        self,
        in_size,
        out_size,
        key=jax.random.PRNGKey(0),
        A=ZeroMatrix(),
        b=ZeroMatrix(),
        C=ZeroMatrix(),
        d=ZeroMatrix(),
    ):
        self.in_size = in_size
        self.out_size = out_size

        keys = jax.random.split(key, 4)
        self.A = (
            A.build(keys[0], (out_size, in_size))
            if isinstance(A, RandomMatrixFactory)
            else A
        )
        self.C = (
            C.build(keys[1], (out_size, in_size))
            if isinstance(C, RandomMatrixFactory)
            else C
        )
        self.b = b.build(keys[2], out_size) if isinstance(b, RandomMatrixFactory) else b
        self.d = d.build(keys[3], out_size) if isinstance(d, RandomMatrixFactory) else d

    @classmethod
    def create_probit(self, in_size, out_size, key=None, A=None, b=None):
        if key is not None:
            keys = jax.random.split(key, 2)
            A = A.build(keys[0], (out_size, in_size))
            b = b.build(keys[1], out_size)
        # int type stops gradient
        C = jnp.zeros((out_size, in_size), dtype=int)
        d = jnp.zeros(out_size, dtype=int)
        return ProbitLinear(in_size, out_size, A=A, b=b, C=C, d=d)

    @classmethod
    def create_residual(self, in_size, out_size, key=None, A=None, b=None):
        if key is not None:
            keys = jax.random.split(key, 2)
            A = A.build(keys[0], (out_size, in_size))
            b = b.build(keys[1], out_size)
        # int type stops gradient
        C = jnp.eye(out_size, dtype=int)
        d = jnp.zeros(out_size, dtype=int)
        return ProbitLinear(in_size, out_size, A=A, b=b, C=C, d=d)

    @classmethod
    def create_linear(self, in_size, out_size, key=None, C=None, d=None):
        if key is not None:
            keys = jax.random.split(key, 2)
            C = C.build(keys[0], (out_size, in_size))
            d = d.build(keys[1], out_size)
        A = jnp.zeros((out_size, in_size), dtype=int)
        b = jnp.zeros(out_size, dtype=int)
        return ProbitLinear(in_size, out_size, A=A, b=b, C=C, d=d)

    @jax.jit
    def __call__(self, x):
        return jax.vmap(σ)(self.A @ x + self.b) + self.C @ x + self.d

    def _propagate_mean(self, μ, Σ):
        return jax.vmap(_M, in_axes=(None, None, 0, 0, 0, 0))(
            μ, Σ, self.A, self.b, self.C, self.d
        )

    def _propagate_cov(self, μ, Σ, rectify=True):
        last_four_axes = (*((None,) * (2 + 3)), *((0,) * 3))
        middle_four_axes = (*((None,) * 2), *((0,) * 3), *((None,) * 3))

        result = jax.vmap(
            jax.vmap(_K, in_axes=middle_four_axes), in_axes=last_four_axes
        )(μ, Σ, self.A, self.b, self.C, self.A, self.b, self.C)
        if rectify:
            result = rectify_eigenvalues(result)
        return result

    def _propagate_mean_lin(self, μ, Σ):
        return self(μ)

    def _propagate_cov_lin(self, μ, Σ):
        J = jax.jacobian(self.__call__)(μ)
        return rectify_eigenvalues(J @ Σ @ J.T)

    def _mc_mean_cov(self, μ, Σ, key, rep):
        input_samples = jax.random.multivariate_normal(key, mean=μ, cov=Σ, shape=rep)
        output_samples = jax.vmap(self.__call__)(input_samples)
        return jnp.mean(output_samples, axis=0), jnp.cov(output_samples.T)

    def _augment_with_identity(self):
        """Returns the network that computes x -> (x, f(x)) where f is this network"""
        A_new = jnp.vstack([jnp.zeros((self.in_size, self.in_size), dtype=int), self.A])
        b_new = jnp.hstack([jnp.zeros(self.in_size, dtype=int), self.b])
        C_new = jnp.vstack([jnp.eye(self.in_size, dtype=int), self.C])
        d_new = jnp.hstack([jnp.zeros(self.in_size, dtype=int), self.d])
        return ProbitLinear(
            in_size=self.in_size,
            out_size=self.in_size + self.out_size,
            A=A_new,
            b=b_new,
            C=C_new,
            d=d_new,
        )

    def _augment_with_sum(self, w_size):
        """Returns the network that computes (w, x) -> (w + f(x)) where f is this network"""
        A_new = jnp.hstack([jnp.zeros((w_size, w_size), dtype=int), self.A])
        b_new = self.b
        C_new = jnp.hstack([jnp.eye(w_size, dtype=int), self.C])
        d_new = self.d
        return ProbitLinear(
            in_size=self.in_size + w_size,
            out_size=self.out_size,
            A=A_new,
            b=b_new,
            C=C_new,
            d=d_new,
        )

    # def _augment_with_proj(self, w_size):
    #     """Returns the network that computes (w, x) -> (w + 0*f(x)) where f is this network"""
    #     A_new = jnp.hstack([jnp.zeros((w_size, w_size), dtype=int), jnp.zeros_like(self.A, dtype=int)])
    #     b_new = jnp.zeros_
    #     C_new = jnp.hstack([jnp.eye(w_size, dtype=int), self.C])
    #     d_new = self.d
    #     return ProbitLinear(
    #         in_size=self.in_size + w_size,
    #         out_size=self.out_size,
    #         A=A_new,
    #         b=b_new,
    #         C=C_new,
    #         d=d_new,
    #     )

    def _direct_sum_with_identity(self, x_size):
        """Returns the network that computes (x, y) -> (x, f(y)) where f is this network"""
        A_new = jax.scipy.linalg.block_diag(
            jnp.zeros((x_size, x_size), dtype=int), self.A
        )
        b_new = jnp.hstack([jnp.zeros(x_size, dtype=int), self.b])
        C_new = jax.scipy.linalg.block_diag(jnp.eye(x_size, dtype=int), self.C)
        d_new = jnp.hstack([jnp.zeros(x_size, dtype=int), self.d])
        return ProbitLinear(
            in_size=x_size + self.in_size,
            out_size=x_size + self.out_size,
            A=A_new,
            b=b_new,
            C=C_new,
            d=d_new,
        )

    def _jitter(self, key, scale):
        keys = jax.random.split(key, 4)
        A_new = (
            self.A + scale * jax.random.normal(keys[0], self.A.shape)
            if equinox.is_inexact_array(self.A)
            else self.A
        )
        b_new = (
            self.b + scale * jax.random.normal(keys[1], self.b.shape)
            if equinox.is_inexact_array(self.b)
            else self.b
        )
        C_new = (
            self.C + scale * jax.random.normal(keys[2], self.C.shape)
            if equinox.is_inexact_array(self.C)
            else self.C
        )
        d_new = (
            self.d + scale * jax.random.normal(keys[3], self.d.shape)
            if equinox.is_inexact_array(self.d)
            else self.d
        )
        return ProbitLinear(
            in_size=self.in_size,
            out_size=self.out_size,
            A=A_new,
            b=b_new,
            C=C_new,
            d=d_new,
        )


class ProbitLinearNetwork(equinox.Module):
    layers: typing.List[ProbitLinear]
    in_size: int
    out_size: int

    def __init__(self, *layers):
        self.layers = layers
        self.in_size = layers[0].in_size
        self.out_size = layers[-1].out_size

    def __getitem__(self, key):
        assert type(key) is int
        return self.layers[key]

    @jax.jit
    def __call__(self, *x):
        x = jnp.concatenate(x)
        for layer in self.layers:
            x = layer(x)
        return x

    @equinox.filter_jit
    def propagate_mean_cov(self, μ, Σ, method="analytic", rectify=True):
        if method == "analytic":
            for layer in self.layers:
                μ, Σ = layer._propagate_mean(μ, Σ), layer._propagate_cov(μ, Σ, rectify)
            return μ, Σ
        elif method == "linear":
            for layer in self.layers:
                μ, Σ = layer._propagate_mean_lin(μ, Σ), layer._propagate_cov_lin(μ, Σ)
            return μ, Σ
        elif method == "unscented":
            μ, Σ = unscented_transform(self.__call__, μ, Σ)
            return μ, rectify_eigenvalues(Σ)
        else:
            raise ValueError(f"propagate_mean_cov: {method} is not a valid method")

    @equinox.filter_jit
    def propagate_mean_cov_block(
        self, means, covariances, method="analytic", rectify=True
    ):
        μ = jnp.concatenate(means)
        Σ = jax.scipy.linalg.block_diag(*covariances)
        return self.propagate_mean_cov(μ, Σ, method, rectify)

    def augment_with_identity(self):
        """Returns the network that computes (x, f(x)) where f(x) is this network"""
        new_layers = [self.layers[0]._augment_with_identity()]
        for layer in self.layers[1:]:
            new_layers.append(layer._direct_sum_with_identity(self.in_size))
        return ProbitLinearNetwork(*new_layers)

    def augment_with_sum(self, w_size):
        """Returns the network that computes (w, x) -> (w + f(x)) where f is this network"""
        new_layers = []
        for layer in self.layers[:-1]:
            new_layers.append(layer._direct_sum_with_identity(w_size))
        new_layers.append(self.layers[-1]._augment_with_sum(w_size))
        return ProbitLinearNetwork(*new_layers)

    def jitter(self, key, scale):
        new_layers = []
        for key, layer in zip(jax.random.split(key, len(self.layers)), self.layers):
            new_layers.append(layer._jitter(key, scale))
        return ProbitLinearNetwork(*new_layers)
