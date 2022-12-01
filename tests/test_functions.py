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
from context import (R, simulation_moments, covariance_matrix, correlation_matrix,
                     _simulation_check, portfolio_cvar, portfolio_var, portfolio_vol)

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


low_risk_pf = np.array([[0.3, 0.1, 0.1, 0.05, 0.225, 0.025, 0.05, 0.05, 0.05, 0.05]]).T
high_risk_pf = np.array([[0.2, 0.1, 0.1, 0.05, 0.275, 0.025, 0.10, 0.05, 0.05, 0.05]]).T
pfs = np.hstack((low_risk_pf, high_risk_pf))

cvar_low = portfolio_cvar(low_risk_pf, R, p1)
cvar_high = portfolio_cvar(high_risk_pf, R, p1)
cvars = portfolio_cvar(pfs, R, p1)
var_low = portfolio_var(low_risk_pf, R, p1)
var_high = portfolio_var(high_risk_pf, R, p1)
vars = portfolio_var(pfs, R, p1)


def test_portfolio_cvar():
    assert cvar_low < cvar_high
    assert np.abs(cvars[0, 0] - cvar_low) <= tol
    assert np.abs(cvars[0, 1] - cvar_high) <= tol


def test_portfolio_var():
    assert var_low < var_high
    assert np.abs(vars[0, 0] - var_low) <= tol
    assert np.abs(vars[0, 1] - var_high) <= tol


def test_var_cvar_relation():
    assert var_low < cvar_low
    assert var_high < cvar_high


def test_var_cvar_raises():
    with pytest.raises(ValueError):
        _ = portfolio_cvar(pfs, R, alpha=1.1)
    with pytest.raises(ValueError):
        _ = portfolio_cvar(pfs, R, alpha='x')
    with pytest.raises(ValueError):
        _ = portfolio_var(pfs, R, demean=1)


def test_portfolio_vol():
    vol_low = portfolio_vol(low_risk_pf, R, p1)
    vol_high = portfolio_vol(high_risk_pf, R, p1)
    vols = portfolio_vol(pfs, R, p1)
    assert vol_low < vol_high
    assert np.abs(vols[0, 0] - vol_low) <= tol
    assert np.abs(vols[0, 1] - vol_high) <= tol
