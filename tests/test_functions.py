# fortitudo.tech - Novel Investment Technologies.
# Copyright (C) 2021-2022 Fortitudo Technologies.

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
import pytest
from context import R, simulation_moments, covariance_matrix, correlation_matrix, _simulation_check

S, I = R.shape
simulation_names = R.columns

p1 = np.ones((S, 1)) / S
p2 = np.random.randint(1, S, (S, 1))
p2 = p2 / np.sum(p2)
tol = 1e-10


@pytest.mark.parametrize("p", [p1, p2])
def test_simulation_moments(p):
    results1 = simulation_moments(R, p)
    results2 = simulation_moments(R.values, p)
    assert results1.shape == (I, 4)
    assert np.all(results1.iloc[:, 1] > 0)
    assert np.all(results1.iloc[:, 3] > 0)
    assert np.all(results1.index == simulation_names)
    assert np.all(results1.values == results2.values)
    assert np.all(results2.index == np.arange(I))


@pytest.mark.parametrize("p", [p1, p2])
def test_covariance_correlation_matrix(p):
    cov1 = covariance_matrix(R, p)
    cov2 = covariance_matrix(R.values, p)
    corr1 = correlation_matrix(R, p)
    corr2 = correlation_matrix(R.values, p)
    assert cov1.shape == (I, I)
    assert np.all(np.diag(cov1) > 0)
    assert np.all(cov1.values == cov2.values)
    assert np.all(corr1.values >= -1 - tol)
    assert np.all(corr1.values <= 1 + tol)
    assert np.all(corr1.values == corr2.values)


def test_simulation_check():
    simulation_names_out1, R_out1, p_out1 = _simulation_check(R, None)
    simulation_names_out2, R_out2, p_out2 = _simulation_check(R.values, p2)
    assert np.all(simulation_names_out1 == simulation_names)
    assert np.all(simulation_names_out2 == np.arange(I))
    assert np.all(R_out1 == R_out2)
    assert np.all(p_out1 == p1)
    assert np.all(p_out2 == p2)
    with pytest.raises(ValueError):
        _ = _simulation_check(R, p1[0:-1])
