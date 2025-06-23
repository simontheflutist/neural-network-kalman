import typing

import equinox
import jax
import scipy.stats
from jax import numpy as jnp


class Normal(equinox.Module):
    μ: jnp.ndarray
    Σ: jnp.ndarray
    n: int

    @staticmethod
    def standard(n):
        return Normal(μ=jnp.zeros(n), Σ=jnp.eye(n))

    @staticmethod
    def certain(μ):
        return Normal(μ=μ, Σ=jnp.zeros((μ.shape[0], μ.shape[0]), dtype=int))

    def __init__(self, μ, Σ, rectify=False):
        self.μ = μ
        self.Σ = Σ if not rectify else rectify_eigenvalues(Σ)
        self.n = μ.shape[0]
        assert self.Σ.shape == (self.n, self.n), self.Σ

    def qmc(self, num_samples, seed=42):
        return scipy.stats.qmc.MultivariateNormalQMC(
            mean=self.μ,
            cov=self.Σ,
            rng=seed,
            engine=scipy.stats.qmc.Sobol(rng=seed, scramble=True, d=self.n),
        ).random(num_samples)

    def samples(self, num_samples, key=42):
        return jax.random.multivariate_normal(
            key, mean=self.μ, cov=self.Σ, shape=num_samples
        )

    def pdf(self, x):
        return scipy.stats.multivariate_normal.pdf(x, mean=self.μ, cov=self.Σ)

    @staticmethod
    def independent(*normals: "Normal") -> "Normal":
        """Creates a joint distribution with zero correlations from multiple Normal distributions."""
        μ = jnp.concatenate([normal.μ for normal in normals])
        Σ_blocks = [normal.Σ for normal in normals]
        Σ = jax.scipy.linalg.block_diag(*Σ_blocks)
        return Normal(μ, Σ)

    def add_covariance(self, cov, at=slice(None, None)):
        return Normal(self.μ, self.Σ.at[at, at].add(cov))

    @equinox.filter_jit
    def __getitem__(self, index: typing.Union[int, slice]):
        """Return the marginal distribution for the specified index."""
        if isinstance(index, int):
            index = slice(index, index + 1)
        if isinstance(index, slice):
            return Normal(self.μ[index], self.Σ[index, index])
        else:
            raise ValueError

    def delete(self, index: int):
        return Normal(
            jnp.delete(self.μ, index),
            jnp.delete(jnp.delete(self.Σ, index, 0), index, 1),
        )

    def condition(
        self, target: slice, given: slice, equals: typing.Union[jnp.ndarray, "Normal"]
    ):
        if isinstance(equals, Normal):
            μ, Σ = schur_complement_2(
                A=self.Σ[target, target],
                B=self.Σ[target, given],
                C=self.Σ[given, given],
                D=equals.Σ,
                x=self.μ[target],
                y=equals.μ - self.μ[given],
            )
        else:
            μ, Σ = schur_complement(
                A=self.Σ[target, target],
                B=self.Σ[target, given],
                C=self.Σ[given, given],
                x=self.μ[target],
                y=equals - self.μ[given],
            )
        return Normal(μ, Σ, rectify=False)

    def χ2(self, x):
        diff = x - self.μ
        return diff.T @ jnp.linalg.solve(self.Σ, diff)


@equinox.filter_jit
def schur_complement(A, B, C, x, y):
    """Returns a numerically stable evaluation of
    x + B C^(-1) y,
    A - B C^(-1) B^T.
    """
    # C = U^T U
    U = jax.scipy.linalg.cholesky(C)
    # B_tilde = B U^-T
    B_tilde = jax.scipy.linalg.solve_triangular(U, B.T, trans=1, lower=False).T
    return (
        x + B_tilde @ jax.scipy.linalg.solve_triangular(U, y, lower=False),
        A - B_tilde.dot(B_tilde.T),
    )


@equinox.filter_jit
def schur_complement_2(A, B, C, D, x, y):
    """Returns a numerically decent evaluation of
    x + G y,
    A + G (D - C) G^T.
    where G = B C^(-1).
    """
    G = jnp.linalg.solve(
        C,
        B.T,
    ).T
    return x + G @ (y), (A + G @ (D - C) @ G.T)


def rectify_eigenvalues(P):
    Λ, V = jnp.linalg.eigh(P, symmetrize_input=1)
    return V @ jnp.diag(jnp.maximum(Λ, 0)) @ V.T
