import typing

import equinox
import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats

import neural_kalman
import normal


class KalmanDiagnostics(equinox.Module):
    kalman_filter: neural_kalman.NeuralKalmanFilter
    x: jnp.ndarray = None
    diagnostic_times: slice = slice(1, None)

    def point_rmse(self, state_trajectory: typing.List[normal.Normal]):
        residuals = self._residuals(state_trajectory)
        return np.mean([(residual**2).sum() for residual in residuals]) ** 0.5

    def _residuals(self, state_trajectory):
        for z, x in zip(
            state_trajectory[self.diagnostic_times], self.x[self.diagnostic_times]
        ):
            yield z.μ[self.kalman_filter.STATES] - x

    def point_geometric_error(self, state_trajectory):
        residuals = self._residuals(state_trajectory)
        outer_product_mean = np.mean(
            [np.outer(residual, residual) for residual in residuals], axis=0
        )
        return np.linalg.det(outer_product_mean) ** (1 / self.kalman_filter.n_x)

    def lpdf(self, state_trajectory):
        return np.mean(
            [
                z.lpdf(x)
                for z, x in zip(
                    state_trajectory[self.diagnostic_times],
                    self.x[self.diagnostic_times],
                )
            ]
        )

    def calculate_coverage(self, predictions: typing.List[normal.Normal]):
        x_pred_χ2 = [
            z.χ2(x)
            for z, x in zip(
                predictions[self.diagnostic_times], self.x[self.diagnostic_times]
            )
        ]
        x_pred_χ2_sorted = sorted(x_pred_χ2)
        x_pred_χ2_theoretical_percentage_points = scipy.stats.chi2(
            self.kalman_filter.n_x
        ).cdf(x_pred_χ2_sorted)
        x_pred_χ2_empirical_percentage_points = scipy.stats.mstats.meppf(
            x_pred_χ2_sorted
        )
        return (
            x_pred_χ2_theoretical_percentage_points,
            x_pred_χ2_empirical_percentage_points,
        )

    def single_coverage(
        self, percentage_point: float, predictions: typing.List[normal.Normal]
    ):
        threshold = scipy.stats.chi2(self.kalman_filter.n_x).ppf(percentage_point)
        covered = [
            z.χ2(x) < threshold
            for z, x in zip(
                predictions[self.diagnostic_times], self.x[self.diagnostic_times]
            )
        ]
        return np.mean(covered)

    def plot_state_trajectories(
        self,
        state_trajectory: typing.List[normal.Normal],
        state_index,
        times=None,
        coverage=0.9,
    ):
        if times is None:
            times = self.diagnostic_times
        plot_times = np.arange(self.x.shape[0])[times]
        plot_state = self.x[times, state_index]
        plot_marginals = [z[state_index] for z in state_trajectory[times]]
        plot_mean = np.array([z.μ for z in plot_marginals]).reshape(-1)

        std = np.array([z.Σ**0.5 for z in plot_marginals]).reshape(-1)
        std_multiplier = scipy.stats.norm.ppf(1 - (1 - coverage) / 2)
        plot_lower = plot_mean - std_multiplier * std
        plot_upper = plot_mean + std_multiplier * std

        covered = np.logical_and(plot_lower <= plot_state, plot_state <= plot_upper)
        missed = np.logical_not(covered)

        return (
            plot_times,
            plot_state,
            plot_mean,
            plot_lower,
            plot_upper,
            covered,
            missed,
        )
