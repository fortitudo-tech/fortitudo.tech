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
from context import R, MeanCVaR, cvar_options, MeanVariance, covariance_matrix

R = R.values
S, I = R.shape
mean = np.mean(R, axis=0)
cov_matrix = np.cov(R, rowvar=False)

G = -np.eye(I)
h = np.zeros(I)
A = np.zeros((1, I))
A[0, 6] = 1
b = np.array([0.1])

tol = 1e-7
np.random.seed(1)
p = np.random.randint(1, S, (S, 1))
p = p / np.sum(p)
mean_random = R.T @ p
cov_matrix_random = covariance_matrix(R, p).values


opt0 = MeanCVaR(R, p=p)
opt1 = MeanVariance(mean_random, cov_matrix_random)


@pytest.mark.parametrize("opt", [(opt0), (opt1)])
def test_long_short(opt):
    min_risk_pf = opt.efficient_portfolio()
    target_return_pf = opt.efficient_portfolio(0.06)
    assert np.abs(np.sum(min_risk_pf) - 1) <= tol
    assert np.abs(np.sum(target_return_pf) - 1) <= tol
    assert np.abs(mean_random.T @ target_return_pf - 0.06) <= tol
    with pytest.raises(ValueError):
        opt.efficient_frontier()


opt2 = MeanCVaR(R, G, h, options={'demean': False})
opt3 = MeanVariance(mean, cov_matrix, G, h)


@pytest.mark.parametrize("opt", [(opt2), (opt3)])
def test_long_only(opt):
    min_risk_lo = opt.efficient_portfolio()
    target_return_lo = opt.efficient_portfolio(0.06)
    assert np.abs(np.sum(min_risk_lo) - 1) <= tol
    assert np.abs(np.sum(target_return_lo) - 1) <= tol
    assert np.abs(np.mean(R @ target_return_lo) - 0.06) <= tol
    frontier_lo = opt.efficient_frontier(4)
    assert frontier_lo.shape == (I, 4)
    assert np.max(np.abs(frontier_lo[:, 0] - min_risk_lo[:, 0])) <= tol


opt4 = MeanCVaR(R, G, h, A, b)
opt5 = MeanVariance(mean, cov_matrix, G, h, A, b)


@pytest.mark.parametrize("opt", [(opt4), (opt5)])
def test_equality_constraint(opt):
    min_risk_eq = opt.efficient_portfolio()
    target_return_eq = opt.efficient_portfolio(0.06)
    assert np.abs(np.sum(min_risk_eq) - 1) <= tol
    assert np.abs(np.sum(target_return_eq) - 1) <= tol
    assert np.abs(np.mean(R @ target_return_eq) - 0.06) <= tol
    assert np.abs(min_risk_eq[6, 0] - b[0]) <= tol
    assert np.abs(target_return_eq[6, 0] - b[0]) <= tol
    frontier_eq = opt.efficient_frontier()
    assert frontier_eq.shape == (I, 9)
    assert np.max(np.abs(frontier_eq[:, 0] - min_risk_eq[:, 0])) <= tol


def test_options():
    cvar_options['demean'] = False
    opt6 = MeanCVaR(R, G, h, A, b)
    assert opt6._demean is False
    with pytest.raises(ValueError):
        MeanCVaR(R, options={'demean': 'X'})
    with pytest.raises(ValueError):
        MeanCVaR(R, options={'R_scalar': -100})
    with pytest.raises(ValueError):
        MeanCVaR(R, options={'maxiter': 'X'})
    with pytest.raises(ValueError):
        MeanCVaR(R, options={'reltol': 1e-9})
    with pytest.raises(ValueError):
        MeanCVaR(R, options={'abstol': 1e-3})


def test_infeasible_constraints():
    G_infeasible = np.vstack((G, -G))
    h_infeasible = np.hstack((h, -np.ones(I)))
    with pytest.raises(ValueError):
        MeanCVaR(R, G=G_infeasible, h=h_infeasible)
    with pytest.raises(ValueError):
        MeanVariance(mean, cov_matrix, G=G_infeasible, h=h_infeasible)


def test_alpha_parameter():
    assert opt0._alpha == 0.95
    opt7 = MeanCVaR(R, G, h, A, b, alpha=0.9)
    assert opt7._alpha == 0.9
    with pytest.raises(ValueError):
        MeanCVaR(R, G, h, A, b, alpha=1.1)
    with pytest.raises(ValueError):
        MeanCVaR(R, G, h, A, b, alpha='x')
