# fortitudo.tech - Novel Investment Technologies.
# Copyright (C) 2021-2025 Fortitudo Technologies.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import pandas as pd
from context import R, exp_decay_probs, normal_exp_decay_calib, covariance_matrix

T, I = R.shape
simulation_names = R.columns
tol = 1e-10

p = exp_decay_probs(R, T / 2)
p1 = exp_decay_probs(R.values, T / 2)
mean, cov_matrix = normal_exp_decay_calib(R, T / 2)
mean1, cov_matrix1 = normal_exp_decay_calib(R.values, T / 2)


def test_exp_decay_probs():
    assert p.shape == (T, 1)
    assert np.abs(np.sum(p) - 1) <= tol
    assert np.abs(p[-int(T / 2 + 1)] / p[-1] - 0.5) <= tol
    assert np.all(np.abs(p - p1)) <= tol


def test_normal_exp_decay_calib():
    assert mean.shape == (I, 1)
    assert cov_matrix.shape == (I, I)
    assert type(mean) is pd.DataFrame
    assert type(cov_matrix) is pd.DataFrame
    assert mean1.shape == (I, 1)
    assert cov_matrix1.shape == (I, I)
    assert type(mean1) is np.ndarray
    assert type(cov_matrix1) is np.ndarray
    assert np.all(np.abs(mean.values - mean1)) <= tol
    assert np.all(np.abs(cov_matrix.values - cov_matrix1)) <= tol


def test_relation():
    assert np.all(np.abs(mean1 - R.values.T @ p)) <= tol
    assert np.all(np.abs(cov_matrix - covariance_matrix(R, p))) <= tol
