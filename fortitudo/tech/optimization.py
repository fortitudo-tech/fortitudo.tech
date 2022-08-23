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
from cvxopt import sparse, matrix
from cvxopt.solvers import lp, qp, options
from typing import Tuple
from copy import copy

options['glpk'] = {'msg_lev': 'GLP_MSG_OFF'}
options['show_progress'] = False
cvar_options = {}


class Optimization:
    def _calculate_max_expected_return(self, feasibility_check: bool = False) -> float:
        """Method for calculating the highest expected return / checking feasibility.

        feasibility_check: Boolean indicating whether the method is used for initial
            feasibility check or to calculate max expected return.

        Returns:
            Highest expected return for the given portfolio constraints.

        Raises:
            ValueError: If constraints are infeasible or max expected return is unbounded.
        """
        if feasibility_check:
            c = matrix(np.zeros(self._G.size[1]))
        else:
            c = self._expected_return_row.T

        solution = lp(c, self._G, self._h, self._A, self._b, solver='glpk')
        if solution['status'] == 'optimal':
            return -solution['primal objective']
        elif feasibility_check:
            raise ValueError('Constraints are infeasible. Please specify feasible constraints.')
        else:
            raise ValueError('Expected return is unbounded. Unable to compute efficient frontier.')

    def efficient_frontier(self, num_portfolios: int = None) -> np.ndarray:
        """Method for computing the efficient frontier.

        Args:
            num_portfolios: Number of portfolios used to span the efficient frontier. Default: 9.

        Returns:
            Efficient frontier with shape (I, num_portfolios).

        Raises:
            ValueError: If expected return is unbounded.
        """
        if num_portfolios is None:
            num_portfolios = 9

        frontier = np.full((self._I, num_portfolios), np.nan)
        frontier[:, 0] = self.efficient_portfolio()[:, 0]

        min_expected_return = float(self._mean @ frontier[:, 0])
        max_expected_return = self._calculate_max_expected_return()
        delta = (max_expected_return - min_expected_return) / (num_portfolios - 1)

        for p in range(1, num_portfolios):
            frontier[:, p] = self.efficient_portfolio(min_expected_return + delta * p)[:, 0]
        return frontier


class MeanCVaR(Optimization):
    """Class for efficient mean-CVaR optimization using Benders decomposition.

    Args:
        R: Matrix with P&L simulations and shape (S, I).
        G: Inequality constraints matrix with shape (N, I).
        h: Inequality constraints vector with shape (N,).
        A: Equality constraints matrix with shape (M, I).
        b: Equality constraints matrix with shape (M,).
        v: Vector of relative market values and shape (I,).
            Default: np.ones(I).
        p: Vector containing scenario probabilities with shape (S, 1).
            Default: np.ones((S, 1)) / S.
        alpha: Alpha value for alpha-VaR and alpha-CVaR. Default: 0.95.
        kwargs: options dictionary with Benders algorithm parameters.

    Raises:
        ValueError: If constraints are infeasible.
    """
    def __init__(
            self, R: np.ndarray, G: np.ndarray = None, h: np.ndarray = None,
            A: np.ndarray = None, b: np.ndarray = None, v: np.ndarray = None,
            p: np.ndarray = None, alpha: float = None, **kwargs: dict):

        self._set_options(kwargs.get('options', globals()['cvar_options']))
        self._S, self._I = R.shape

        if v is None:
            self._v = np.hstack((np.ones((1, self._I)), np.zeros((1, 2))))
        else:
            self._v = np.hstack((v[np.newaxis, :], np.zeros((1, 2))))

        if G is None:
            self._G = sparse(matrix(np.hstack((np.zeros((1, self._I + 1)), [[-1]]))))
            self._h = matrix([0.])
        else:
            self._G = sparse(matrix(
                np.block([[G, np.zeros((G.shape[0], 2))], [np.zeros(self._I + 1), -1]])))
            self._h = matrix(np.hstack((h, [0.])))

        if A is None:
            self._A = sparse(matrix(self._v))
            self._b = matrix([1.])
        else:
            self._A = sparse(matrix(np.block([[A, np.zeros((A.shape[0], 2))], [self._v]])))
            self._b = matrix(np.hstack((b, [1.])))

        _ = self._calculate_max_expected_return(feasibility_check=True)

        if p is None:
            self._p = np.ones((1, self._S)) / self._S
        else:
            self._p = p.T

        if alpha is None:
            self._alpha = 0.95
        elif type(alpha) is float and 0 < alpha < 1:
            self._alpha = alpha
        else:
            raise ValueError('alpha must be a float in the interval (0, 1).')
        self._c = matrix(np.hstack((np.zeros(self._I), [1], [1 / (1 - self._alpha)])))

        self._mean = self._p @ R
        self._expected_return_row = matrix(np.hstack((-self._mean, np.zeros((1, 2)))))
        if self._demean:
            self._losses = -self._R_scalar * (R - self._mean)
        else:
            self._losses = -self._R_scalar * R

    def _set_options(self, options: dict):
        """Method for setting Benders algorithm parameters.

        Args:
            options: Dictionary containing algorithm parameters.
        """
        self._demean = options.get('demean', True)
        if type(self._demean) != bool:
            raise ValueError('demean must be a boolean equal to True or False.')
        self._R_scalar = options.get('R_scalar', 1000)
        if type(self._R_scalar) not in (int, float) or self._R_scalar <= 0:
            raise ValueError('R_scalar must be a postive integer or float.')
        self._maxiter = options.get('maxiter', 500)
        if type(self._maxiter) != int or self._maxiter < 100:
            raise ValueError('maxiter must be a postive integer greater than or equal to 100.')
        self._reltol = options.get('reltol', 1e-8)
        if not 1e-8 <= self._reltol <= 1e-4:
            raise ValueError('reltol must be in [1e-8, 1e-4].')
        self._abstol = options.get('abstol', 1e-8)
        if not 1e-8 <= self._abstol <= 1e-4:
            raise ValueError('abstol must be in [1e-8, 1e-4].')

    def _benders_algorithm(self, G: sparse, h: matrix) -> np.ndarray:
        """Method for running Benders algorithm.

        Args:
            G: Inequality constraints matrix with shape (N, I) or (N+1, I).
            h: Inequality constraints vector with shape (N, 1) or (N+1, I).

        Returns:
            Solution to the mean-CVaR optimization problem.
        """
        eta = self._p @ self._losses
        p = 1
        solution, w, F_lower, G_benders, h_benders, eta, p = self._benders_main(G, h, eta, p)
        F_star = F_lower + self._c[-1] * (w - solution[-1])
        v = 1
        while self._benders_stopping_criteria(F_star, F_lower) and v <= self._maxiter:
            solution, w, F_lower, G_benders, h_benders, eta, p = self._benders_main(
                G_benders, h_benders, eta, p)
            F_star = min(F_lower + self._c[-1] * (w - solution[-1]), F_star)
            v += 1
        return solution

    def _benders_main(
            self, G_benders: sparse, h_benders: matrix, eta: np.ndarray, p: float
            ) -> Tuple[np.ndarray, float, float, sparse, matrix, np.ndarray, float]:
        """Method for solving the current relaxed master problem and updating cut.

        Args:
            G_benders: Matrix containing inequality constraints and cuts.
            h_benders: Vector containing inequality constraints and cuts.
            eta: Average loss for the current cut.
            p: Sum of probabilities for the current cut.

        Returns:
            Current solution as well as cut and stopping criteria variables.
        """
        G_benders = sparse([G_benders, matrix(np.block([eta, -p, -1]))])
        h_benders = matrix([h_benders, 0])
        solution = np.array(
            lp(c=self._c, G=G_benders, h=h_benders, A=self._A, b=self._b, solver='glpk')['x'])
        eta, p = self._benders_cut(solution)
        w = eta @ solution[0:-2] - p * solution[-2]
        F_lower = self._c.T @ solution
        return solution, w, F_lower, G_benders, h_benders, eta, p

    def _benders_cut(self, solution: np.ndarray) -> Tuple[np.ndarray, float]:
        """Method for generating Benders cut.

        Args:
            solution: Current solution.

        Returns:
            Input for the next cut.
        """
        K = (self._losses @ solution[0:self._I] >= solution[-2])[:, 0]
        eta = self._p[:, K] @ self._losses[K, :]
        p = np.sum(self._p[0, K])
        return eta, p

    def _benders_stopping_criteria(self, F_star: float, F_lower: float) -> bool:
        """Method for assessing if the algorithm should continue.

        Args:
            F_star: Best upper bound on the objective value.
            F_lower: Current lower bound on the objective value.

        Returns:
            Boolean indicating whether the algorithm should continue.
        """
        F_lower_abs = np.abs(F_lower)
        if F_lower_abs > 1e-10:
            return (F_star - F_lower) / F_lower_abs > self._reltol
        else:
            return (F_star - F_lower) > self._abstol

    def efficient_portfolio(self, return_target: float = None) -> np.ndarray:
        """Method for computing a mean-CVaR efficient portfolio with return a target.

        Args:
            return_target: Return target for the efficient portfolio.
                The minimum CVaR portfolio is computed by default.

        Returns:
            Efficient portfolio exposures with shape (I, 1).
        """
        if return_target is None:
            G = copy(self._G)
            h = copy(self._h)
        else:
            G = sparse([self._G, self._expected_return_row])
            h = matrix([self._h, -return_target])
        return self._benders_algorithm(G, h)[0:-2]


class MeanVariance(Optimization):
    """Class for efficient mean-variance optimization.

    Args:
        mean: Mean vector with shape (I,).
        covariance_matrix: Covariance matrix with shape (I, I).
        G: Inequality constraints matrix with shape (N, I).
        h: Inequality constraints vector with shape (N,).
        A: Equality constraints matrix with shape (M, I).
        b: Equality constraints matrix with shape (M,).
        v: Vector of relative market values and shape (I,).
            Default: np.ones(I).

    Raises:
        ValueError: If constraints are infeasible.
    """
    def __init__(
            self, mean: np.ndarray, covariance_matrix: np.ndarray,
            G: np.ndarray = None, h: np.ndarray = None, A: np.ndarray = None,
            b: np.ndarray = None, v: np.ndarray = None):

        self._I = len(mean)
        self._mean = mean
        self._expected_return_row = -matrix(mean).T
        self._P = matrix(covariance_matrix)
        self._q = matrix(np.zeros(self._I))

        if v is None:
            self._v = np.ones((1, self._I))
        else:
            self._v = v[np.newaxis, :]

        if G is None:
            self._G = sparse(matrix(np.zeros((1, self._I))))
            self._h = matrix([0.])
        else:
            self._G = sparse(matrix(G))
            self._h = matrix(h)

        if A is None:
            self._A = sparse(matrix(self._v))
            self._b = matrix([1.])
        else:
            self._A = sparse(matrix(np.vstack((A, self._v))))
            self._b = matrix(np.hstack((b, [1.])))

        _ = self._calculate_max_expected_return(feasibility_check=True)

    def efficient_portfolio(self, return_target: float = None) -> np.ndarray:
        """Method for computing a mean-variance efficient portfolio with a return target.

        Args:
            return_target: Return target for the efficient portfolio.
                The minimum variance portfolio is computed by default.

        Returns:
            Efficient portfolio exposures with shape (I, 1).
        """
        if return_target is None:
            return np.array(qp(self._P, self._q, self._G, self._h, self._A, self._b)['x'])
        else:
            G = sparse([self._G, self._expected_return_row])
            h = matrix([self._h, -return_target])
            return np.array(qp(self._P, self._q, G, h, self._A, self._b)['x'])
