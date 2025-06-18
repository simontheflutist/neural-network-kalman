import equinox
import jax
import numpy as np
from jax import numpy as jnp

from probit_network import ProbitLinearNetwork


class NeuralKalmanFilter(equinox.Module):
    n_x: int
    n_u: int
    n_y: int
    F: ProbitLinearNetwork
    F_aug: ProbitLinearNetwork
    H_aug: ProbitLinearNetwork
    Q: jnp.ndarray
    R: jnp.ndarray
    STATES: slice
    NEXT_STATES: slice
    INPUTS: slice
    OUTPUTS: slice
    JOINT: slice

    def __init__(self, n_x, n_u, n_y, F, H, Q, R):
        assert F.out_size == n_x, "F.out_size != n_x"
        assert F.in_size == n_x + n_u, "F.in_size != n_x + n_u"
        assert H.out_size == n_y, "H.out_size != n_y"
        assert H.in_size == n_x + n_u, "H.in_size != n_x + n_u"

        self.n_x = n_x
        self.n_u = n_u
        self.n_y = n_y
        self.F = F
        self.H_aug = H.augment_with_identity()
        self.Q = Q
        self.R = R

        self.STATES = slice(None, n_x)
        self.INPUTS = slice(n_x, n_x + n_u)
        self.OUTPUTS = slice(-n_y, None)
        self.JOINT = slice(None, n_x + n_u + n_y)

        # Used for smoothing
        self.F_aug = F.augment_with_identity()
        self.NEXT_STATES = slice(n_x, None)

    @equinox.filter_jit
    def predict(self, x, P, method="analytic"):
        """Predicts the next state and output given the current state."""
        # predict state
        x, P = self.F.propagate_mean_cov(x, P, method=method, rectify=True)
        P = P + self.Q
        # predict joint distribution of state and output
        x_and_y, P_x_and_y = self.H_aug.propagate_mean_cov(
            x, P, method=method, rectify=True
        )
        P_x_and_y = P_x_and_y.at[self.OUTPUTS, self.OUTPUTS].add(self.R)
        return x_and_y, P_x_and_y

    @equinox.filter_jit
    def predict_with_input(self, x, u, P, method="analytic"):
        """Predicts the next state and output distribution given the current state and exogenous input."""
        # predict state
        x_and_u, P_x_and_u = self.F.propagate_mean_cov_block(
            (x, u), (P, jnp.zeros((self.n_u, self.n_u))), method=method, rectify=True
        )
        P_x = P_x_and_u[self.STATES, self.STATES] + self.Q
        # predict joint distribution of state and output
        x_and_u_and_y, P_x_and_u_and_y = self.H_aug.propagate_mean_cov_block(
            (x_and_u[self.STATES], u),
            (P_x, jnp.zeros((self.n_u, self.n_u))),
            method=method,
            rectify=True,
        )
        P_x_and_u_and_y = P_x_and_u_and_y.at[self.OUTPUTS, self.OUTPUTS].add(self.R)
        # discard the input
        x_and_y = jnp.delete(x_and_u_and_y, self.INPUTS)
        P_x_and_y = jnp.delete(
            jnp.delete(P_x_and_u_and_y, self.INPUTS, 0), self.INPUTS, 1
        )
        return x_and_y, P_x_and_y

    @equinox.filter_jit
    def correct(self, x_and_y, P_x_and_y, y):
        x, P = NeuralKalmanFilter.schur_complement(
            P_x_and_y[self.STATES, self.STATES],
            P_x_and_y[self.STATES, self.OUTPUTS],
            P_x_and_y[self.OUTPUTS, self.OUTPUTS],
            x_and_y[self.STATES],
            y - x_and_y[self.OUTPUTS],
        )
        return x, P

    @equinox.filter_jit
    def correct_with_recalibrate(
        self,
        x_and_y,
        P_x_and_y,
        y,
        method="analytic",
        backout="trace",
        return_recalibration_difference=False,
    ):
        x = x_and_y[self.STATES]
        P_x = P_x_and_y[self.STATES, self.STATES]
        S = P_x_and_y[self.OUTPUTS, self.OUTPUTS]
        K = jnp.linalg.solve(S, P_x_and_y[self.OUTPUTS, self.STATES]).T
        x_updated = x + K @ (y - x_and_y[self.OUTPUTS])
        # In a regular KF, we would return x_updated along with an updated covariance matrix
        # given by a Schur complement expression.  In the recalibrated KF, we return either
        #     0. x (no update) and P_x (no update), or
        #     1. x_updated and P_x_recal
        # whichever has the "larger" P according to some total ordering on PSD matrices compatible
        # with the Loewner ordering.
        # P_x_recal comes from a formula that uses the same Kalman gain K as in the regular KF,
        #  but evaluates other terms at x_updated rather than x.
        #
        # This follows Jiang et al. 2024 "A new framework for nonlinear Kalman filters"
        _, P_x_and_y_recal = self.H_aug.propagate_mean_cov(
            x_updated, P_x, method=method, rectify=True
        )
        P_x_and_y_recal = P_x_and_y_recal.at[self.OUTPUTS, self.OUTPUTS]
        S_recal = P_x_and_y_recal[self.OUTPUTS, self.OUTPUTS]
        P_x_recal = (
            P_x_and_y[self.STATES, self.STATES]
            + K @ S_recal @ K.T
            - P_x_and_y_recal[self.STATES, self.OUTPUTS] @ K.T
            - K @ P_x_and_y_recal[self.OUTPUTS, self.STATES]
        )

        # back out: switch between P_x and P_x_recal
        if backout == "trace":
            backout_criterion = P_x_recal.trace() > P_x.trace()
        elif backout == "det":
            backout_criterion = (
                jnp.linalg.slogdet(P_x_recal)[1] > jnp.linalg.slogdet(P_x)[1]
            )
        elif backout == "always":
            backout_criterion = True
        elif backout == "never":
            backout_criterion = False
        else:
            raise ValueError

        # if backout criterion is true, return the un-updated x and P_x
        x = jnp.where(backout_criterion, x, x_updated)
        P = jnp.where(backout_criterion, P_x, P_x_recal)

        if return_recalibration_difference:
            return x, P, jnp.linalg.norm(P_x_and_y_recal - P_x_and_y)
        return x, P

    @equinox.filter_jit
    def smooth(self, x_current, P_current, x_next, P_next, method="analytic"):
        # joint distribution of x_current and F(x_current)
        x_current_and_next, P_current_and_next = self.F_aug.propagate_mean_cov(
            x_current, P_current, method=method
        )
        # smoothing gain
        G = jnp.linalg.solve(
            P_current_and_next[self.NEXT_STATES, self.NEXT_STATES] + self.Q,
            P_current_and_next[self.NEXT_STATES, self.STATES],
        ).T
        # smoothing update
        x_smoothed = x_current + G @ (x_next - x_current_and_next[self.NEXT_STATES])
        P_smoothed = (
            P_current
            + G
            @ (P_next - P_current_and_next[self.NEXT_STATES, self.NEXT_STATES])
            @ G.T
        )
        return x_smoothed, P_smoothed

    @staticmethod
    @equinox.filter_jit
    def schur_complement(A, B, C, x, y):
        """Returns a numerically stable(ish) attempt at
        x + B C^(-1) y,
        A - B C^(-1) B^T.
        """
        # C = U U^T
        U = jax.scipy.linalg.cholesky(C)
        # B_tilde = B U^-T
        B_tilde = jax.scipy.linalg.solve_triangular(U, B.T, trans=1, lower=False).T
        return (
            x + B_tilde @ jax.scipy.linalg.solve_triangular(U, y, lower=False),
            A - B_tilde.dot(B_tilde.T),
        )
