import equinox
from jax import numpy as jnp


class Normal(equinox.Module):
    μ: jnp.ndarray
    Σ: jnp.ndarray
