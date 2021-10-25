# fortitudo.tech investment and risk technologies
# Copyright (C) 2021 Fortitudo Technologies ApS

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
    if G is None:
        solution = minimize(
            _dual_equality, x0=np.zeros(A.shape[0]), args=(p, A, b),
            method='Newton-CG', jac=True, hess=_hessian_equality,
            options={'maxiter': 10000})
        q = np.exp(np.log(p) - 1 - A.T @ solution.x[:, np.newaxis])
    else:
        len_b = len(b)
        len_h = len(h)
        len_bh = len_b + len_h
        bounds = Bounds([-np.inf] * len_b + [0] * len_h, [np.inf] * len_bh)
        solution = minimize(
            _dual_inequality, x0=np.zeros(len_bh), args=(p, A, b, G, h),
            method='TNC', jac=True, bounds=bounds, options={'maxiter': 10000})
        q = np.exp(np.log(p) - 1
                   - A.T @ solution.x[0:len_b][:, np.newaxis]
                   - G.T @ solution.x[len_b:][:, np.newaxis])
    return q


def _dual_equality(
        equality_multipliers: np.ndarray, p: np.ndarray, A: np.ndarray,
        b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Function computing equality constrained objective and gradient.

    Args:
        equality_multipliers: Lagrange multipliers with shape (M,).
        p: Prior probability vector with shape (S, 1).
        A: Equality constraint matrix with shape (M, S).
        b: Equality constraint vector with shape (M, 1).

    Returns:
        Tuple containing the dual objective value and gradient.
    """
    equality_multipliers = equality_multipliers[:, np.newaxis]
    x = np.exp(np.log(p) - 1 - A.T @ equality_multipliers)
    dual_objective = (x.T @ (np.log(x) - np.log(p))
                      + equality_multipliers.T @ (A @ x - b))
    gradient = b - A @ x
    return -100 * dual_objective, gradient.flatten()


def _dual_inequality(
        lagrange_multipliers: np.ndarray, p: np.ndarray, A: np.ndarray, b: np.ndarray,
        G: np.ndarray, h: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Function computing inequality constrained objective and gradient.

    Args:
        lagrange_multipliers: Lagrange multipliers with shape (N + M,).
        p: Prior probability vector with shape (S, 1).
        A: Equality constraint matrix with shape (M, S).
        b: Equality constraint vector with shape (M, 1).
        G: Inequality constraint matrix with shape (N, S).
        h: Inequality constraint vector with shape (N, 1).

    Returns:
        Tuple containing the dual objective value and gradient.
    """
    equality_multipliers = lagrange_multipliers[0:len(b)][:, np.newaxis]
    inequality_multipliers = lagrange_multipliers[len(b):][:, np.newaxis]
    x = np.exp(np.log(p) - 1
               - A.T @ equality_multipliers
               - G.T @ inequality_multipliers)
    dual_objective = (x.T @ (np.log(x) - np.log(p))
                      + equality_multipliers.T @ (A @ x - b)
                      + inequality_multipliers.T @ (G @ x - h))
    gradient = np.vstack((b - A @ x, h - G @ x))
    return -100 * dual_objective, gradient


def _hessian_equality(
        equality_multipliers: np.ndarray, p: np.ndarray,
        A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Function computing equality constrained hessian.

    Args:
        equality_multipliers: Lagrange multipliers with shape (M,).
        p: Prior probability vector with shape (S, 1).
        A: Equality constraint matrix with shape (M, S).
        b: Equality constraint vector with shape (M, 1).

    Returns:
        Hessian matrix with shape (M, M).
    """
    equality_multipliers = equality_multipliers[:, np.newaxis]
    x = np.exp(np.log(p) - 1 - A.T @ equality_multipliers)
    hessian = A @ (x @ np.ones((1, A.shape[0])) * A.T)
    return hessian
