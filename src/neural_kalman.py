import equinox
import jax
from jax import numpy as jnp

import normal
from normal import Normal
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
    def predict(self, x: Normal, method="analytic"):
        """Predicts the next state and output given the current state."""
        # predict state
        x_pred = self.F(x, method=method, rectify=True).add_covariance(self.Q)
        # predict joint distribution of state and output
        x_and_y_pred = self.H_aug(x_pred, method=method, rectify=True).add_covariance(
            self.R, at=self.OUTPUTS
        )
        return x_and_y_pred

    @equinox.filter_jit
    def predict_with_input(self, x, u, method="analytic"):
        """Predicts the next state and output distribution given the current state and exogenous input."""
        # predict state
        x_and_u = Normal.independent(x, u)
        x_pred = self.F(x_and_u, method=method, rectify=True).add_covariance(self.Q)
        # predict joint distribution of state and output
        x_pred_and_u = Normal.independent(x_pred, u)
        x_and_y_pred = self.H_aug(
            x_pred_and_u, method=method, rectify=True
        ).add_covariance(self.R, at=self.OUTPUTS)
        # discard the input
        return x_and_y_pred.delete(self.INPUTS)

    @equinox.filter_jit
    def correct(
        self,
        x_and_y,
        y,
        recalibrate=False,
        recalibrate_method="analytic",
        recalibrate_backout="trace",
    ):
        if recalibrate:
            raise NotImplementedError
            return self._recalibrated_correct(
                x_and_y,
                y,
                method=recalibrate_method,
                backout=recalibrate_backout,
            )
        else:
            return x_and_y.condition(self.STATES, given=self.OUTPUTS, equals=y)

    @equinox.filter_jit
    def _recalibrated_correct(
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
        P_x_and_y_recal = P_x_and_y_recal.at[self.OUTPUTS, self.OUTPUTS].add(self.R)
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
    def smooth(self, x_current: Normal, x_next: Normal, method="analytic"):
        # joint distribution of x_current and F(x_current)
        x_current_and_next = self.F_aug(x_current, method=method).add_covariance(
            self.Q, at=self.NEXT_STATES
        )
        # condition on F(x_current) = x_next
        return x_current_and_next.condition(
            target=self.STATES, given=self.NEXT_STATES, equals=x_next
        )

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
