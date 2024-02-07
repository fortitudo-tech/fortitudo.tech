# fortitudo.tech - Novel Investment Technologies.
# Copyright (C) 2021-2024 Fortitudo Technologies.

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
from scipy.optimize import minimize, Bounds
from typing import Tuple


def entropy_pooling(
        p: np.ndarray, A: np.ndarray, b: np.ndarray, G: np.ndarray = None,
        h: np.ndarray = None, method: str = None) -> np.ndarray:
    """Function for computing Entropy Pooling posterior probabilities.

    Args:
        p: Prior probability vector with shape (S, 1).
        A: Equality constraint matrix with shape (M, S).
        b: Equality constraint vector with shape (M, 1).
        G: Inequality constraint matrix with shape (N, S).
        h: Inequality constraint vector with shape (N, 1).
        method: Optimization method: {'TNC', 'L-BFGS-B'}. Default 'TNC'.

    Returns:
        Posterior probability vector with shape (S, 1).
    """
    if method is None:
        method = 'TNC'
    elif method not in ('TNC', 'L-BFGS-B'):
        raise ValueError(f'Method {method} not supported. Choose TNC or L-BFGS-B.')

    len_b = len(b)
    if G is None:
        lhs = A
        rhs = b
        bounds = Bounds([-np.inf] * len_b, [np.inf] * len_b)
    else:
        lhs = np.vstack((A, G))
        rhs = np.vstack((b, h))
        len_h = len(h)
        bounds = Bounds([-np.inf] * len_b + [0] * len_h, [np.inf] * (len_b + len_h))

    log_p = np.log(p)
    dual_solution = minimize(
        _dual_objective, x0=np.zeros(lhs.shape[0]), args=(log_p, lhs, rhs),
        method=method, jac=True, bounds=bounds, options={'maxfun': 10000})
    q = np.exp(log_p - 1 - lhs.T @ dual_solution.x[:, np.newaxis])
    return q


def _dual_objective(
        lagrange_multipliers: np.ndarray, log_p: np.ndarray,
        lhs: np.ndarray, rhs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Function computing Entropy Pooling dual objective and gradient.

    Args:
        lagrange_multipliers: Lagrange multipliers with shape (M,) or (M + N,).
        log_p: Log of prior probability vector with shape (S, 1).
        lhs: Matrix with shape (M, S) or (M + N, S).
        rhs: Vector with shape (M, 1) or (M + N, 1).

    Returns:
        Dual objective value and gradient.
    """
    lagrange_multipliers = lagrange_multipliers[:, np.newaxis]
    log_x = log_p - 1 - lhs.T @ lagrange_multipliers
    x = np.exp(log_x)
    gradient = rhs - lhs @ x
    objective = x.T @ (log_x - log_p) - lagrange_multipliers.T @ gradient
    return -1000 * objective, 1000 * gradient
