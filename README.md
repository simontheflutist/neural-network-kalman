# Neural Network Kalman Filtering (AI summary of docs/nn-filtering.pdf)

This repository focuses on the application of neural networks in the context of nonlinear Kalman filtering, smoothing, and prediction. The main highlights of the work include the following:

## Overview

The document introduces a general Kalman filter algorithm for recursive prediction and filtering, as well as a general RTS smoother algorithm for recursive smoothing. The approach is motivated by Bayesian principles and aims to unify different types of nonlinear Kalman filters.

## Algorithms

- **Kalman Filter**: A recursive algorithm for prediction and filtering, which is presented in a stylized manner to emphasize its Bayesian motivation.
- **RTS Smoother**: An algorithm for recursive smoothing, which builds upon the Kalman filter output.

## Mathematical Framework

- **State Dynamics**: Described using a highly stylized notation to depict stochastic processes and their realizations.
- **Uncertainty Propagation**: Emphasizes the propagation of uncertainty using specific operators and the distinction between marginalizing and conditionalizing operations.

## Computational Complexity

The document reports on computational complexity using JAX ahead-of-time compilation, focusing on input-output operations and floating-point operations. Comparisons are provided for filtering and smoothing methods.

## Numerical Example

- **Stochastic Lorenz System**: A nonlinear Kalman filtering application on a discretization of the stochastic Lorenz system is demonstrated.
- **Parameterization and Training**: The neural network architecture consists of four hidden layers, with specific configurations for constant-covariance generative models.

## Results

Monte Carlo simulations are performed to evaluate filtering and smoothing performance, using various methods and baselining against a dynamic model with stationary distribution predictions.

This repository is an exploration of advanced filtering techniques using neural networks, providing insights into both theoretical and practical aspects of nonlinear stochastic systems. For more details, please refer to the original document and accompanying code.

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