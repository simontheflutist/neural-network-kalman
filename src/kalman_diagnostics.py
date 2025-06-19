import equinox
import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats

import neural_kalman


class KalmanDiagnostics(equinox.Module):
    kalman_filter: neural_kalman.NeuralKalmanFilter
    x: jnp.ndarray = None
    diagnostic_times: slice = slice(1, None)

    def calculate_coverage(self, predicted_mean, predicted_covariance):
        x_pred_χ2 = jax.vmap(
            lambda residual, cov: residual @ jnp.linalg.solve(cov, residual)
        )(
            predicted_mean[self.diagnostic_times, :] - self.x[self.diagnostic_times, :],
            predicted_covariance[
                self.diagnostic_times,
                self.kalman_filter.STATES,
                self.kalman_filter.STATES,
            ],
        )
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

    def plot_state_trajectories(
        self, predicted_mean, predicted_covariance, state_index, times, coverage=0.9
    ):
        plot_times = np.arange(self.x.shape[0])[times]
        plot_state = self.x[times, state_index]
        plot_mean = predicted_mean[times, self.kalman_filter.STATES][:, state_index]

        std = np.sqrt(
            predicted_covariance[
                times, self.kalman_filter.STATES, self.kalman_filter.STATES
            ][:, state_index, state_index]
        )
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
