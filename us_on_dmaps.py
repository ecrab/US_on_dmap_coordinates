import numpy as np
from typing import Tuple

import argparse
import os
import math

from sklearn.gaussian_process import GaussianProcessRegressor

L: float = 8.0                  # Interval length
N: int = 20                     # Number of sampled points
n: int = 20000                  # Number of evaluation points
    

def gpr_gradient(gpr: GaussianProcessRegressor, Xstar: np.ndarray) -> np.ndarray:
    """Evaluate the gradient of a Gaussian process at a given point."""
    X = gpr.X_train_
    m, n = Xstar.shape[0], X.shape[0]
    assert Xstar.shape[1] == X.shape[1]
    Xstar_minus_X = Xstar[:, np.newaxis, :] - X[np.newaxis, :, :]
    kXstarX = gpr.kernel_(Xstar, X)
    length_scale = gpr.kernel_.get_params()['k2__length_scale']
    kXstarX_times_alpha = (kXstarX * gpr.alpha_)[:, :, np.newaxis]
    return -(kXstarX_times_alpha * Xstar_minus_X).sum(axis=1) / length_scale**2 * gpr._y_train_std

def US(gpr: GaussianProcessRegressor, n, kappa, cv):
        epsilon: float = 1e-3           # Inverse temperature.
        num_steps: int = n              # Number of steps to integrate.
        dim: int = 2                    # Spatial dimensions.
        dt: float = 1                # Time step length.
        k = kappa                       # Spring constant
        cond_var = cv                    # Conditioned variable
        def vector_field(x: np.ndarray) -> np.ndarray:
            sample = np.array((x[0], x[1]))
            sample = np.array(sample.reshape(1,-1))
            grad = gpr_gradient(gpr, sample)
            pred = gpr.predict(sample)
            vector = np.array([1e-3, (1-x[1])*1e-1])
            bias = k*(pred-cond_var)*grad
            biased_vec = vector - bias
            return biased_vec
        scale_param = np.array([1e-3, 1e-1])
        random_number_generator = np.random.default_rng()
        xs = np.zeros((num_steps, dim))
        xs[0,1] = 1
        xs[0,0] = 0
        for i in range(1, num_steps):
            x_prev = xs[i-1, :]
            noise = random_number_generator.normal(scale=scale_param, size=2)
            xs[i, :] = x_prev + vector_field(x_prev) * dt + noise
        return xs
