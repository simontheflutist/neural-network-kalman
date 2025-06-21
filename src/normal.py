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
        return Normal(jnp.zeros(n), jnp.eye(n))

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

    def __add__(self, other):
        if isinstance(other, Normal):
            new_μ = jnp.concatenate([self.μ, other.μ])
            new_Σ = jax.scipy.linalg.block_diag(self.Σ, other.Σ)
            return Normal(new_μ, new_Σ)
        raise NotImplementedError(
            "Addition is only supported with another Normal distribution."
        )

    def __getitem__(self, index: typing.Union[int, slice]):
        """Return the marginal distribution for the specified index."""
        if isinstance(index, int):
            index = slice(index, index + 1)
        if isinstance(index, slice):
            return Normal(self.μ[index], self.Σ[index, index])
        else:
            raise ValueError


def rectify_eigenvalues(P):
    Λ, V = jnp.linalg.eigh(P, symmetrize_input=1)
    return V @ jnp.diag(jnp.maximum(Λ, 0)) @ V.T
