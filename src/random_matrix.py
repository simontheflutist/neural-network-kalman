import abc

import equinox
import jax
import jax.numpy as jnp


class RandomMatrixFactory(equinox.Module):
    @abc.abstractmethod
    def build(self, key, shape):
        pass


class ConstantMatrix(RandomMatrixFactory):
    value: float

    def __init__(self, value=1):
        self.value = value

    def build(self, key, shape):
        return jnp.full(shape, self.value)


class ZeroMatrix(RandomMatrixFactory):
    def build(self, key, shape):
        return jnp.zeros(shape)


class RandomGaussian(RandomMatrixFactory):
    scale: float

    def __init__(self, scale=1):
        self.scale = scale

    @equinox.filter_jit
    def build(self, key, shape):
        if type(shape) is int:
            scale = 1
        else:
            scale = 1 / jnp.sqrt(shape[0])
        return jax.random.normal(key, shape=shape) * self.scale * scale


class RandomUniform(RandomMatrixFactory):
    min_val: float
    max_val: float

    def __init__(self, min_val=-jnp.pi, max_val=jnp.pi):
        self.min_val = min_val
        self.max_val = max_val

    @equinox.filter_jit
    def build(self, key, shape):
        return jax.random.uniform(
            key, shape=shape, minval=self.min_val, maxval=self.max_val
        )


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
