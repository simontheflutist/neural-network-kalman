# Neural Network Kalman Filtering
This project deals with Gaussian filtering and smoothing of dynamic systems described by

$$\begin{align}
x_{t} &= F(x_{t-1}, u_{t}) + \epsilon_t\\
y_t &= H(x_t, u_t) + \eta_t
\end{align}
$$
where $F$ and $H$ are deep neural networks consisting of layers
$$
f(x; A, b, C, d) = \sigma(Ax + b) + Cx + d.
$$

By taking $\sigma(x) = 2 \Phi(x) - 1$, we can compute the mean and covariance of $f(x)$ exactly.
Details in `docs/nn-filtering.pdf`.

# Library (AI summary of `src/`)

The library  is organized as follows:

* `neural_kalman.py`: Implements the Kalman filter and RTS smoother algorithms, as well as the associated prediction and update steps.
* `kalman_diagnostics.py`: Provides functions for evaluating the performance of the algorithms.
* `probit_network.py`: Defines the neural network architecture used in the numerical examples.
* `random_matrix.py`: Implements random matrix generators for initializing neural networks.

# MVP example: filtering and smoothing the stochastic Lorenz system
Implemented:
- solving the Lorenz system
- profile maximum likelihood training of a discrete transition model with constant process covariance
- Kalman filtering using [analytic|linear|unscented] uncertainty propagation with support for Jiang et al.~(2025) update step.
- diagnostics for both point estimation (RMSE) and coverage ($\chi^2$ confidence ellipsoids)