import abc

import equinox
import jax
import jax.numpy as jnp


class RandomMatrixFactory(equinox.Module):
    @abc.abstractmethod
    def build(self, key, shape):
        pass


class ZeroMatrix(RandomMatrixFactory):
    def build(self, key, shape):
        return jnp.zeros(shape)


class RandomGaussian(RandomMatrixFactory):
    scale: float

    def __init__(self, scale=1):
        self.scale = scale

    @equinox.filter_jit
    def build(self, key, shape):
        return jax.random.normal(key, shape=shape) * self.scale


class RandomOrthogonalProjection(RandomMatrixFactory):
    scale: float

    def __init__(self, scale=1):
        self.scale = scale

    @equinox.filter_jit
    def build(self, key, shape):
        if type(shape) is int:
            Z = jax.random.normal(key, shape)
            return self.scale * Z / jnp.linalg.norm(Z)
        N = max(shape)
        Z = jax.random.normal(key, (N, N))
        U, _, V_H = jnp.linalg.svd(Z, full_matrices=False, compute_uv=True)
        return self.scale * U[: shape[0], :] @ V_H[:, : shape[1]]
