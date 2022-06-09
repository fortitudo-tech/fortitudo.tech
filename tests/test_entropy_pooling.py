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
from context import entropy_pooling, R

R = R.values
S = len(R)
A_base = np.ones((1, S))
b_base = np.ones((1, 1))
A = np.vstack((A_base, R[:, 0][np.newaxis, :]))
b = np.vstack((b_base, np.array([[0.075]])))
G = -R[:, 1][np.newaxis, :]
h = -np.array([[0.075]])
tol = 1e-5
p1 = np.ones((S, 1)) / S
p2 = np.random.randint(1, S, (S, 1))
p2 = p2 / np.sum(p2)


@pytest.mark.parametrize("p", [p1, p2])
def test_equality(p):
    q = entropy_pooling(p, A, b)
    means = q.T @ R
    assert np.abs(means[0, 0] - b[1, 0]) <= tol
    assert np.abs(np.sum(q) - 1) <= tol
    assert np.all(q > 0)
    assert q.shape == (S, 1)


@pytest.mark.parametrize("p", [p1, p2])
def test_base_inequality(p):
    q = entropy_pooling(p, A_base, b_base, G, h)
    means = q.T @ R
    assert means[0, 1] + h[0, 0] <= tol
    assert np.abs(np.sum(q) - 1) <= tol
    assert np.all(q > 0)
    assert q.shape == (S, 1)


@pytest.mark.parametrize("p", [p1, p2])
def test_equality_inequality(p):
    q = entropy_pooling(p, A, b, G, h)
    means = q.T @ R
    assert np.abs(means[0, 0] - b[1, 0]) <= tol
    assert means[0, 1] + h[0, 0] <= tol
    assert np.abs(np.sum(q) - 1) <= tol
    assert np.all(q > 0)
    assert q.shape == (S, 1)
