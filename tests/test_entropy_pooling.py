# fortitudo.tech investment and risk technologies
# Copyright (C) 2021 Fortitudo Technologies ApS

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from context import entropy_pooling, pnl

S = len(pnl)
A_base = np.ones((1, S))
b_base = np.ones((1, 1))
A = np.vstack((A_base, pnl[:, 0][np.newaxis, :]))
b = np.vstack((b_base, np.array([[0.075]])))
G = -pnl[:, 1][np.newaxis, :]
h = -np.array([[0.075]])
tol = 1e-5


def test_uniform_prior():
    p = np.ones((S, 1)) / S
    q1 = entropy_pooling(p, A, b, G, h)
    q2 = entropy_pooling(p, A, b)
    q3 = entropy_pooling(p, A_base, b_base, G, h)
    means1 = q1.T @ pnl
    means2 = q2.T @ pnl
    means3 = q3.T @ pnl
    assert np.abs(means1[0, 0] - b[1, 0]) <= tol
    assert means1[0, 1] + h[0, 0] <= tol
    assert np.abs(np.sum(q1) - 1) <= tol
    assert np.all(q1 > 0)
    assert q1.shape == (S, 1)
    assert np.abs(means2[0, 0] - b[1, 0]) <= tol
    assert np.abs(np.sum(q2) - 1) <= tol
    assert np.all(q2 > 0)
    assert q2.shape == (S, 1)
    assert means3[0, 1] + h[0, 0] <= tol
    assert np.abs(np.sum(q3) - 1) <= tol
    assert np.all(q3 > 0)
    assert q3.shape == (S, 1)


def test_random_prior():
    p2 = np.random.randint(1, S, (S, 1))
    p2 = p2 / np.sum(p2)
    q4 = entropy_pooling(p2, A, b, G, h)
    q5 = entropy_pooling(p2, A, b)
    q6 = entropy_pooling(p2, A_base, b_base, G, h)
    means4 = q4.T @ pnl
    means5 = q5.T @ pnl
    means6 = q6.T @ pnl
    assert np.abs(means4[0, 0] - b[1, 0]) <= tol
    assert means4[0, 1] + h[0, 0] <= tol
    assert np.abs(np.sum(q4) - 1) <= tol
    assert np.all(q4 > 0)
    assert q4.shape == (S, 1)
    assert np.abs(means5[0, 0] - b[1, 0]) <= tol
    assert np.abs(np.sum(q5) - 1) <= tol
    assert np.all(q5 > 0)
    assert q5.shape == (S, 1)
    assert means6[0, 1] + h[0, 0] <= tol
    assert np.abs(np.sum(q6) - 1) <= tol
    assert np.all(q6 > 0)
    assert q6.shape == (S, 1)
