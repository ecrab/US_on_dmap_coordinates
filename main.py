import numpy as np
from typing import Tuple

import argparse
import os
import math

import scipy.stats as stats
import scipy.spatial
from scipy.spatial.distance import pdist
from sklearn.gaussian_process import GaussianProcessRegressor

import traj_sim
import dmaps
import us_on_dmaps

x_real, y_real = traj_sim.generate_real_samples(20000)

# x_real = np.load('./Data/paper_untransformed_xs.npy')
# y_real = np.load('./Data/paper_untransformed_labels.npy')

pw_dists = pdist(x_real, "euclidean")
eps_par = np.median(pw_dists**2)*1
print(eps_par)

eigenvalues, eigenvectors = dmaps.diffusion_maps(x_real, eps_par)
first_eigv = -1*eigenvectors[:, 0]

X, y = x_real, first_eigv

gpr = GaussianProcessRegressor()
gpr.fit(X, y)
assert np.allclose(gpr.score(X, y), 1.0), 'Fitted GP is not good enough'


samples = us_on_dmaps.US(gpr, 5000, 5e9, 0.0000)