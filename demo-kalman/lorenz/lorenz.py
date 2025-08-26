from dataclasses import dataclass

import diffrax
import equinox
import jax.numpy as jnp


@dataclass
class LorenzArgs:
    """Dataclass to hold Lorenz arguments."""

    σ: float = 1e-3
    T: float = 1.0
    dt: float = 1e-2


class LorenzSDE(equinox.Module):
    σ: float
    T: float
    dt: float
    tol: float
    solver_args: dict
    sigma: float
    rho: float
    beta: float

    def __init__(self, lorenz_args, solver_args=None, tol=None):
        self.σ = lorenz_args.σ
        self.T = lorenz_args.T
        self.dt = lorenz_args.dt
        self.tol = tol or self.dt / 2
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
            **self.solver_args
        )
        return sol if return_sol else sol.ys

    @equinox.filter_jit
    def F_full(self, x, key, num_save):
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
            saveat=diffrax.SaveAt(ts=jnp.linspace(0, self.T, num_save, endpoint=True)),
            **self.solver_args
        )

        return sol.ts, sol.ys
