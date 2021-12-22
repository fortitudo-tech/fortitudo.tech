# fortitudo.tech - Novel Investment Technologies.
# Copyright (C) 2021 Fortitudo Technologies ApS.

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
from context import MeanCVaR, MeanVariance, cvar_options, pnl

S, I = pnl.shape
mean = np.mean(pnl, axis=0)
covariance_matrix = np.cov(pnl, rowvar=False)
tol = 1e-7
G = -np.eye(I)
h = np.zeros(I)
A = np.zeros((2, I))
A[0, 6] = 1
A[1, :] = 1
b = np.array([0.1, 1])


def test_long_short():
    np.random.seed(1)
    p = np.random.randint(1, S, (S, 1))
    p = p / np.sum(p)
    opt = MeanCVaR(pnl, p=p)
    min_risk_pf = opt.efficient_portfolio()
    target_return_pf = opt.efficient_portfolio(0.06)
    assert np.abs(np.sum(min_risk_pf) - 1) <= tol
    assert np.abs(np.sum(target_return_pf) - 1) <= tol
    assert np.abs(p.T @ pnl @ target_return_pf - 0.06) <= tol
    with pytest.raises(ValueError):
        opt.efficient_frontier()


def test_long_only():
    opt2 = MeanCVaR(pnl, G=G, h=h)
    min_risk_lo = opt2.efficient_portfolio()
    target_return_lo = opt2.efficient_portfolio(0.06)
    assert np.abs(np.sum(min_risk_lo) - 1) <= tol
    assert np.abs(np.sum(target_return_lo) - 1) <= tol
    assert np.abs(np.mean(pnl @ target_return_lo) - 0.06) <= tol
    frontier_lo = opt2.efficient_frontier(4)
    assert frontier_lo.shape == (I, 4)
    assert np.max(np.abs(frontier_lo[:, 0] - min_risk_lo[:, 0])) <= tol


def test_equality_constraint():
    cvar_options['demean'] = False
    opt3 = MeanCVaR(pnl, A, b, G, h)
    min_risk_eq = opt3.efficient_portfolio()
    target_return_eq = opt3.efficient_portfolio(0.06)
    assert np.abs(np.sum(min_risk_eq) - 1) <= tol
    assert np.abs(np.sum(target_return_eq) - 1) <= tol
    assert np.abs(np.mean(pnl @ target_return_eq) - 0.06) <= tol
    assert np.abs(min_risk_eq[6, 0] - b[0]) <= tol
    assert np.abs(target_return_eq[6, 0] - b[0]) <= tol
    frontier_eq = opt3.efficient_frontier()
    assert frontier_eq.shape == (I, 9)
    assert np.max(np.abs(frontier_eq[:, 0] - min_risk_eq[:, 0])) <= tol


def test_long_short_variance():
    opt4 = MeanVariance(mean, covariance_matrix)
    min_risk_pf = opt4.efficient_portfolio()
    target_return_pf = opt4.efficient_portfolio(0.06)
    assert np.abs(np.sum(min_risk_pf) - 1) <= tol
    assert np.abs(np.sum(target_return_pf) - 1) <= tol
    assert np.abs(mean @ target_return_pf - 0.06) <= tol
    with pytest.raises(ValueError):
        opt4.efficient_frontier()


def test_long_only_varaince():
    opt5 = MeanVariance(mean, covariance_matrix, G=G, h=h)
    min_risk_lo = opt5.efficient_portfolio()
    target_return_lo = opt5.efficient_portfolio(0.06)
    assert np.abs(np.sum(min_risk_lo) - 1) <= tol
    assert np.abs(np.sum(target_return_lo) - 1) <= tol
    assert np.abs(np.mean(pnl @ target_return_lo) - 0.06) <= tol
    frontier_lo = opt5.efficient_frontier(4)
    assert frontier_lo.shape == (I, 4)
    assert np.max(np.abs(frontier_lo[:, 0] - min_risk_lo[:, 0])) <= tol


def test_equality_constraint_variance():
    opt6 = MeanVariance(mean, covariance_matrix, A, b, G, h)
    min_risk_eq = opt6.efficient_portfolio()
    target_return_eq = opt6.efficient_portfolio(0.06)
    assert np.abs(np.sum(min_risk_eq) - 1) <= tol
    assert np.abs(np.sum(target_return_eq) - 1) <= tol
    assert np.abs(np.mean(pnl @ target_return_eq) - 0.06) <= tol
    assert np.abs(min_risk_eq[6, 0] - b[0]) <= tol
    assert np.abs(target_return_eq[6, 0] - b[0]) <= tol
    frontier_eq = opt6.efficient_frontier()
    assert frontier_eq.shape == (I, 9)
    assert np.max(np.abs(frontier_eq[:, 0] - min_risk_eq[:, 0])) <= tol


def test_inputs():
    with pytest.raises(ValueError):
        MeanCVaR(pnl, G=G)
    with pytest.raises(ValueError):
        MeanCVaR(pnl, h=h)
    with pytest.raises(ValueError):
        MeanCVaR(pnl, A=A)
    with pytest.raises(ValueError):
        MeanCVaR(pnl, b=b)


def test_options():
    with pytest.raises(ValueError):
        MeanCVaR(pnl, options={'demean': 'X'})
    with pytest.raises(ValueError):
        MeanCVaR(pnl, options={'R_scalar': -100})
    with pytest.raises(ValueError):
        MeanCVaR(pnl, options={'maxiter': 'X'})
    with pytest.raises(ValueError):
        MeanCVaR(pnl, options={'reltol': 1e-9})
    with pytest.raises(ValueError):
        MeanCVaR(pnl, options={'abstol': 1e-3})
