"""adapted from https://github.com/jax-ml/jax/issues/10562"""

import jax
import jax.numpy as jnp
from jax.scipy.special import erf
from jax.scipy.stats.norm import cdf as cdf1d


def case1(p, q, rho, a, b):
    c1 = -1.0950081470333
    c2 = -0.75651138383854
    sqrt2 = jnp.sqrt(2)
    a2 = a * a
    b2 = b * b

    line11 = 0.5 * (erf(q / sqrt2) + erf(b / (sqrt2 * a)))
    aux1 = a2 * c1 * c1 - 2 * sqrt2 * b * c1 + 2 * b2 * c2
    aux2 = 4 * (1 - a2 * c2)
    line12 = (1.0 / (4 * jnp.sqrt(1 - a2 * c2))) * jnp.exp(aux1 / aux2)

    line21 = 1 - erf((sqrt2 * b - a2 * c1) / (2 * a * jnp.sqrt(1 - a2 * c2)))
    aux3 = a2 * c1 * c1 + 2 * sqrt2 * b * c1 + 2 * b2 * c2
    line22 = (1.0 / (4 * jnp.sqrt(1 - a2 * c2))) * jnp.exp(aux3 / aux2)

    aux4 = sqrt2 * q - sqrt2 * a2 * c2 * q - sqrt2 * a * b * c2 - a * c1
    aux5 = 2 * jnp.sqrt(1 - a2 * c2)
    line31 = erf(aux4 / aux5)
    line32 = erf((a2 * c1 + sqrt2 * b) / (2 * a * jnp.sqrt(1 - a2 * c2)))

    return line11 + (line12 * line21) - (line22 * (line31 + line32))


def case2(p, q):
    return cdf1d(p) * cdf1d(q)


def case3(p, q, rho, a, b):
    c1 = -1.0950081470333
    c2 = -0.75651138383854
    sqrt2 = jnp.sqrt(2)
    a2 = a * a
    b2 = b * b

    aux1 = a2 * c1 * c1 - 2 * sqrt2 * b * c1 + 2 * b2 * c2
    aux2 = 4 * (1 - a2 * c2)
    line11 = (1.0 / (4 * jnp.sqrt(1 - a2 * c2))) * jnp.exp(aux1 / aux2)

    aux3 = sqrt2 * q - sqrt2 * a2 * c2 * q - sqrt2 * a * b * c2 + a * c1
    aux4 = 2 * jnp.sqrt(1 - a2 * c2)
    line12 = 1.0 + erf(aux3 / aux4)

    return line11 * line12


def case4(p, q, rho, a, b):
    c1 = -1.0950081470333
    c2 = -0.75651138383854
    sqrt2 = jnp.sqrt(2)
    a2 = a * a
    b2 = b * b

    line11 = 0.5 + 0.5 * erf(q / sqrt2)
    aux1 = a2 * c1 * c1 + 2 * sqrt2 * b * c1 + 2 * b2 * c2
    aux2 = 4 * (1 - a2 * c2)
    line12 = (1.0 / (4 * jnp.sqrt(1 - a2 * c2))) * jnp.exp(aux1 / aux2)

    aux3 = sqrt2 * q - sqrt2 * a2 * c2 * q - sqrt2 * a * b * c2 - a * c1
    aux4 = 2 * jnp.sqrt(1 - a2 * c2)
    line2 = 1.0 + erf(aux3 / aux4)

    return line11 - (line12 * line2)


def case5(p, q, rho, a, b):
    c1 = -1.0950081470333
    c2 = -0.75651138383854
    sqrt2 = jnp.sqrt(2)
    a2 = a * a
    b2 = b * b

    line11 = 0.5 - 0.5 * erf(b / (sqrt2 * a))
    aux1 = a2 * c1 * c1 + 2 * sqrt2 * b * c1 + 2 * b2 * c2
    aux2 = 4 * (1 - a2 * c2)
    line12 = (1.0 / (4 * jnp.sqrt(1 - a2 * c2))) * jnp.exp(aux1 / aux2)

    line21 = 1 - erf((sqrt2 * b + a2 * c1) / (2 * a * jnp.sqrt(1 - a2 * c2)))
    aux3 = a2 * c1 * c1 - 2 * sqrt2 * b * c1 + 2 * b2 * c2
    line22 = (1.0 / (4 * jnp.sqrt(1 - a2 * c2))) * jnp.exp(aux3 / aux2)

    aux4 = sqrt2 * q - sqrt2 * a2 * c2 * q - sqrt2 * a * b * c2 + a * c1
    aux5 = 2 * jnp.sqrt(1 - a2 * c2)
    line31 = erf(aux4 / aux5)
    line32 = erf((-a2 * c1 + sqrt2 * b) / (2 * a * jnp.sqrt(1 - a2 * c2)))

    return line11 - (line12 * line21) + line22 * (line31 + line32)


@jax.jit
def binorm(x1, x2, rho):
    p = x1
    q = x2

    a = -rho / jnp.sqrt(1 - rho * rho)
    b = p / jnp.sqrt(1 - rho * rho)

    cond1 = (a > 0) & (a * q + b >= 0)
    cond2 = a == 0
    cond3 = (a > 0) & (a * q + b < 0)
    cond4 = (a < 0) & (a * q + b >= 0)

    index = jnp.select([cond1, cond2, cond3, cond4], [0, 1, 2, 3], default=4)

    def case0(_):
        return case1(p, q, rho, a, b)

    def case1_(_):
        return case2(p, q)

    def case2_(_):
        return case3(p, q, rho, a, b)

    def case3_(_):
        return case4(p, q, rho, a, b)

    def case4_(_):
        return case5(p, q, rho, a, b)

    cases = [case0, case1_, case2_, case3_, case4_]

    return jax.lax.switch(index, cases, operand=None)
