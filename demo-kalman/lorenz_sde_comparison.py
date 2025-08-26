#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from enum import Enum, auto

import equinox
import jax
import numpy as np
import scipy.stats
from jax import numpy as jnp
from matplotlib.figure import Figure

# JAX setup
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Local imports: resolve src relative to this file
_HERE = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.append(os.path.join(_REPO_ROOT, "src"))
import activation
import network
import normal
import random_matrix
from unscented import UnscentedTransformMethod


class LorenzSDE(equinox.Module):
    σ: float
    T: float
    dt: float
    tol: float
    solver_args: dict
    sigma: float
    rho: float
    beta: float

    def __init__(self, σ=2.0, T=0.05, dt=1e-2, solver_args=None, tol=None):
        self.σ = σ
        self.T = T
        self.dt = dt
        self.tol = tol or dt / 2
        self.solver_args = solver_args or {}

        # Fixed Lorenz parameters
        self.sigma = 10.0
        self.rho = 28.0
        self.beta = 8.0 / 3.0

    def drift(self, t, y, args=None):
        x, y_, z = y
        dx = self.sigma * (y_ - x)
        dy = x * (self.rho - z) - y_
        dz = x * y_ - self.beta * z
        return jnp.array([dx, dy, dz])

    def diffusion(self, t, y, args=None):
        return jnp.eye(3) * self.σ

    def make_bm(self, key):
        return diffrax.VirtualBrownianTree(
            t0=0,
            t1=self.T,
            tol=self.tol,
            shape=(3,),
            key=key,
            levy_area=diffrax.SpaceTimeLevyArea,
        )

    def make_sde(self, bm):
        return diffrax.MultiTerm(
            diffrax.ODETerm(self.drift),
            diffrax.ControlTerm(self.diffusion, bm),
        )

    @equinox.filter_jit
    def F(self, x, key, return_sol=False):
        bm = self.make_bm(key)
        sde = self.make_sde(bm)
        solver = diffrax.ShARK()
        sol = diffrax.diffeqsolve(
            sde,
            solver=solver,
            t0=0,
            t1=self.T,
            dt0=self.dt,
            y0=x,
            max_steps=10000,
            **self.solver_args,
        )
        return sol if return_sol else sol.ys


class Method(Enum):
    LINEAR = auto()
    UNSCENTED95 = auto()
    UNSCENTED02 = auto()
    MEAN_FIELD = auto()
    ANALYTIC = auto()


def build_lorenz_network(in_size=3, hidden=60, key=jax.random.PRNGKey(0)):
    """Build the Lorenz surrogate network with the same topology as used in training.
    The weights are placeholders; we'll load real weights if available.
    """
    keys = jax.random.split(key, 6)
    act = activation.Sinusoid()

    A_in = random_matrix.RandomGaussian(scale=1.0)
    b_in = random_matrix.RandomUniform()

    layers = [
        network.Layer.create_nonlinear(
            in_size=in_size,
            out_size=hidden,
            activation=act,
            key=keys[0],
            A=A_in,
            b=b_in,
        ),
        network.Layer.create_residual(
            in_size=hidden,
            out_size=hidden,
            activation=act,
            key=keys[1],
            A=random_matrix.RandomOrthogonalProjection(),
            b=random_matrix.RandomUniform(),
        ),
        network.Layer.create_residual(
            in_size=hidden,
            out_size=hidden,
            activation=act,
            key=keys[2],
            A=random_matrix.RandomOrthogonalProjection(),
            b=random_matrix.RandomUniform(),
        ),
        network.Layer.create_residual(
            in_size=hidden,
            out_size=hidden,
            activation=act,
            key=keys[3],
            A=random_matrix.RandomOrthogonalProjection(),
            b=random_matrix.RandomUniform(),
        ),
        network.Layer.create_residual(
            in_size=hidden,
            out_size=hidden,
            activation=act,
            key=keys[4],
            A=random_matrix.RandomOrthogonalProjection(),
            b=random_matrix.RandomUniform(),
        ),
        network.Layer.create_linear(
            in_size=hidden,
            out_size=in_size,
            key=keys[5],
            C=random_matrix.ZeroMatrix(),
            d=random_matrix.ZeroMatrix(),
        ),
    ]
    return network.Network(*layers)


@dataclass
class LorenzSDEComparison:
    """Test-case style comparison for Lorenz SDE surrogate UQ methods."""

    sde: LorenzSDE
    f_net: network.Network
    input_mean: jnp.ndarray
    input_cov: jnp.ndarray
    num_samples: int = 2**9  # 512 for speed
    num_repetitions: int = 3
    output_dir: str | None = None

    def __post_init__(self):
        self.dist = normal.Normal(self.input_mean, self.input_cov)
        # Output path (match test_case style), resolved from repo root unless overridden
        self.base_path = (
            self.output_dir
            if self.output_dir is not None
            else os.path.join(_REPO_ROOT, "docs", "manuscript", "generated")
        )
        os.makedirs(self.base_path, exist_ok=True)
        os.makedirs(os.path.join(self.base_path, "distributions"), exist_ok=True)

    def run(self):
        logger.info("Generating QMC inputs and true/network outputs")
        with jax.default_device(jax.devices("cpu")[0]):
            # Repetitions for uncertainty on scalar metrics if needed later
            mc_inputs = [
                self.dist.qmc(self.num_samples, seed=i)
                for i in range(self.num_repetitions)
            ]

            true_outputs = [
                jax.vmap(self.sde.F)(
                    mc_inputs[i],
                    jax.random.split(jax.random.PRNGKey(1000 + i), self.num_samples),
                ).squeeze()
                for i in range(self.num_repetitions)
            ]
            net_outputs = [
                jax.vmap(self.f_net)(mc_inputs[i]) for i in range(self.num_repetitions)
            ]

            # Flatten all repetitions for pseudo and plotting
            true_flat = np.array(true_outputs).reshape(-1, self.input_mean.shape[0])
            net_flat = np.array(net_outputs).reshape(-1, self.input_mean.shape[0])

            # Empirical normals
            dist_true = normal.Normal.from_samples(true_flat)
            dist_net = normal.Normal.from_samples(net_flat)

            # Approximations
            approx = {
                Method.LINEAR: self.f_net(self.dist, method="linear"),
                Method.UNSCENTED0: self.f_net(
                    self.dist,
                    method="unscented",
                    unscented_method=UnscentedTransformMethod.UT0_VECTOR,
                ),
                Method.UNSCENTED1: self.f_net(
                    self.dist,
                    method="unscented",
                    unscented_method=UnscentedTransformMethod.UT1_VECTOR,
                ),
                Method.ANALYTIC: self.f_net(self.dist, method="analytic"),
            }

        # Print summary metrics
        logger.info("Summary KL divergences (true -> approx)")
        for name in Method:
            kl = dist_true.kl_divergence(approx[name]).item()
            logger.info(f"KL(true || {name.name.lower()}) = {kl:.3e}")

        logger.info("Summary KL divergences (net -> approx)")
        for name in Method:
            kl = dist_net.kl_divergence(approx[name]).item()
            logger.info(f"KL(net || {name.name.lower()}) = {kl:.3e}")

        # Plot per-dimension histograms with Gaussian overlays (like test_case)
        dist_path = self.plot_distributions(net_flat, dist_true, approx)
        logger.info(f"Saved distribution comparison to {dist_path}")
        return dist_path

    def plot_distributions(
        self, samples: np.ndarray, dist_true: normal.Normal, approx: dict
    ):
        y_dim = samples.shape[1]
        y_mesh = []
        for i in range(y_dim):
            mu_i = float(jnp.atleast_1d(dist_true[i].μ).reshape(-1)[0])
            var_i = float(jnp.atleast_2d(dist_true[i].Σ).reshape(-1)[0])
            sd_i = var_i**0.5
            lo = min(mu_i - 3 * sd_i, float(np.percentile(samples[:, i], 0.5)))
            hi = max(mu_i + 3 * sd_i, float(np.percentile(samples[:, i], 99.5)))
            y_mesh.append(np.linspace(lo, hi, 2000).reshape(-1))

        fig = Figure(dpi=300, figsize=(6, 6), constrained_layout=1)
        for i in range(y_dim):
            ax = fig.add_subplot(2, 2, i + 1)
            ax.hist(
                samples[:, i], bins=50, density=True, alpha=0.5, label="empirical (net)"
            )

            # Overlays
            ax.plot(
                y_mesh[i],
                jax.vmap(lambda y: dist_true[i].pdf(jnp.array([y])))(
                    jnp.array(y_mesh[i])
                ),
                label="pseudo-GT",
                linestyle="--",
            )
            ax.plot(
                y_mesh[i],
                jax.vmap(lambda y: approx[Method.ANALYTIC][i].pdf(jnp.array([y])))(
                    jnp.array(y_mesh[i])
                ),
                label="analytic",
                linestyle="-",
            )
            ax.plot(
                y_mesh[i],
                jax.vmap(lambda y: approx[Method.LINEAR][i].pdf(jnp.array([y])))(
                    jnp.array(y_mesh[i])
                ),
                label="linear",
                linestyle="-",
            )
            ax.plot(
                y_mesh[i],
                jax.vmap(lambda y: approx[Method.UNSCENTED0][i].pdf(jnp.array([y])))(
                    jnp.array(y_mesh[i])
                ),
                label="unscented'95",
                linestyle="-.",
            )
            ax.plot(
                y_mesh[i],
                jax.vmap(lambda y: approx[Method.UNSCENTED1][i].pdf(jnp.array([y])))(
                    jnp.array(y_mesh[i])
                ),
                label="unscented'02",
                linestyle=":",
            )
            ax.set_title(f"y[{i+1}]")
            ax.legend(fontsize=6)

        out_path = os.path.join(
            self.base_path, "distributions", "lorenz_sde_comparison.pdf"
        )
        fig.savefig(out_path)
        return out_path


def main():
    parser = argparse.ArgumentParser(description="Lorenz SDE UQ comparison")
    parser.add_argument("--samples", type=int, default=2**9)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=1e-2)
    parser.add_argument("--sde-sigma", type=float, default=1e-3)
    parser.add_argument("--eqx", type=str, default="")
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--seed", type=int, default=1312)
    args = parser.parse_args()

    # SDE setup
    sde = LorenzSDE(SDE_sigma=args.sde_sigma, T=args.T, dt=args.dt)

    # Surrogate network
    f_net = build_lorenz_network(
        in_size=3, hidden=60, key=jax.random.PRNGKey(args.seed)
    )

    # Try loading trained weights if available
    if args.eqx:
        candidates = [args.eqx]
    else:
        candidates = [
            os.path.join(_REPO_ROOT, c) for c in ("lorenz.eqx", "lorenz-20k.eqx")
        ]
    for candidate in candidates:
        if os.path.exists(candidate):
            logger.info(f"Loading network weights from {candidate}")
            try:
                f_net = equinox.tree_deserialise_leaves(candidate, f_net)
                break
            except Exception as e:
                logger.warning(f"Failed to load {candidate}: {e}")

    # Input distribution (centered at a typical Lorenz state)
    x0 = jnp.array([-8.0, 4.0, 27.0])
    Q = (0.5**2) * jnp.eye(3)  # moderate uncertainty

    out_dir = os.path.abspath(args.out) if args.out else None
    comp = LorenzSDEComparison(
        sde=sde,
        f_net=f_net,
        input_mean=x0,
        input_cov=Q,
        num_samples=args.samples,
        num_repetitions=args.reps,
        output_dir=out_dir,
    )
    comp.run()


if __name__ == "__main__":
    main()
