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
from scipy.optimize import minimize, Bounds
from typing import Tuple


def entropy_pooling(
        p: np.ndarray, A: np.ndarray, b: np.ndarray,
        G: np.ndarray = None, h: np.ndarray = None) -> np.ndarray:
    """Function for computing Entropy Pooling posterior probabilities.

    Args:
        p: Prior probability vector with shape (S, 1).
        A: Equality constraint matrix with shape (M, S).
        b: Equality constraint vector with shape (M, 1).
        G: Inequality constraint matrix with shape (N, S).
        h: Inequality constraint vector with shape (N, 1).

    Returns:
        Posterior probability vector with shape (S, 1).
    """
    log_p = np.log(p)
    if G is None:
        solution = minimize(
            _dual_equality, x0=np.zeros(A.shape[0]), args=(log_p, A, b),
            method='Newton-CG', jac=True, hess=_hessian_equality,
            options={'maxiter': 10000})
        q = np.exp(log_p - 1 - A.T @ solution.x[:, np.newaxis])
    else:
        len_b = len(b)
        len_h = len(h)
        len_bh = len_b + len_h
        bounds = Bounds([-np.inf] * len_b + [0] * len_h, [np.inf] * len_bh)
        lhs = np.vstack((A, G))
        solution = minimize(
            _dual_inequality, x0=np.zeros(len_bh), args=(log_p, lhs, np.vstack((b, h))),
            method='TNC', jac=True, bounds=bounds, options={'maxiter': 10000})
        q = np.exp(log_p - 1 - lhs.T @ solution.x[:, np.newaxis])
    return q


def _dual_equality(
        equality_multipliers: np.ndarray, log_p: np.ndarray, A: np.ndarray,
        b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Function computing equality constrained objective and gradient.

    Args:
        equality_multipliers: Lagrange multipliers with shape (M,).
        log_p: Log of prior probability vector with shape (S, 1).
        A: Equality constraint matrix with shape (M, S).
        b: Equality constraint vector with shape (M, 1).

    Returns:
        Tuple containing the dual objective value and gradient.
    """
    equality_multipliers = equality_multipliers[:, np.newaxis]
    log_x = log_p - 1 - A.T @ equality_multipliers
    x = np.exp(log_x)
    gradient = b - A @ x
    dual_objective = x.T @ (log_x - log_p) - equality_multipliers.T @ gradient
    return -dual_objective, gradient.flatten()


def _dual_inequality(
        lagrange_multipliers: np.ndarray, log_p: np.ndarray, lhs: np.ndarray,
        rhs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Function computing inequality constrained objective and gradient.

    Args:
        lagrange_multipliers: Lagrange multipliers with shape (M + N,).
        log_p: Log of prior probability vector with shape (S, 1).
        lhs: Matrix with shape (M + N, S).
        rhs: Vector with shape (M + N, 1).

    Returns:
        Tuple containing the dual objective value and gradient.
    """
    lagrange_multipliers = lagrange_multipliers[:, np.newaxis]
    log_x = log_p - 1 - lhs.T @ lagrange_multipliers
    x = np.exp(log_x)
    gradient = rhs - lhs @ x
    dual_objective = x.T @ (log_x - log_p) - lagrange_multipliers.T @ gradient
    return -1000 * dual_objective, 1000 * gradient


def _hessian_equality(
        equality_multipliers: np.ndarray, log_p: np.ndarray,
        A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Function computing equality constrained hessian.

    Args:
        equality_multipliers: Lagrange multipliers with shape (M,).
        log_p: Log of prior probability vector with shape (S, 1).
        A: Equality constraint matrix with shape (M, S).
        b: Equality constraint vector with shape (M, 1).

    Returns:
        Hessian matrix with shape (M, M).
    """
    x = np.exp(log_p - 1 - A.T @ equality_multipliers[:, np.newaxis])
    hessian = A @ (x @ np.ones((1, A.shape[0])) * A.T)
    return hessian
