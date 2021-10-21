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
from cvxopt import sparse, matrix, solvers
from typing import Tuple
from copy import copy

solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF'}
cvar_options = {}


class MeanCVaR:
    """Object for efficient mean-CVaR optimization using Benders decomposition.

    Args:
        R: Matrix with P&L simulations and shape (S, I).
        A: Equality constraints matrix with shape (M, I).
        b: Equality constraints matrix with shape (M,).
        G: Inequality constraints matrix with shape (N, I).
        h: Inequality constraints vector with shape (N,).
        p: Vector containing scenario probabilities with shape (S, 1).
            Default: np.ones((S, 1)) / S.
        alpha: Alpha value for alpha-VaR and alpha-CVaR. Default: 0.95.
        kwargs: options dictionary with CVaR algorithm parameters.
    """
    def __init__(
            self, R: np.ndarray, A: np.ndarray = None, b: np.ndarray = None,
            G: np.ndarray = None, h: np.ndarray = None, p: np.ndarray = None,
            alpha: float = None, **kwargs: dict):

        self.S, self.I = R.shape
        if A is not None and b is not None:
            self.A = sparse(matrix(np.hstack((A, np.zeros((A.shape[0], 2))))))
            self.b = matrix(b)
        elif A is None and b is None:
            self.A = sparse(matrix(np.hstack((np.ones((1, self.I)), np.zeros((1, 2))))))
            self.b = matrix([1.])
        else:
            raise ValueError('A and b must both be None or both different from None.')

        if G is not None and h is not None:
            self.G = sparse(matrix(
                np.block([[G, np.zeros((G.shape[0], 2))], [np.zeros(self.I + 1), -1]])))
            self.h = matrix(np.hstack((h, [0.])))
        elif G is None and h is None:
            self.G = sparse(matrix(np.hstack((np.zeros((1, self.I + 1)), [[-1]]))))
            self.h = matrix([0.])
        else:
            raise ValueError('G and h must both be None or both different from None.')

        if p is None:
            self.p = np.ones((1, self.S)) / self.S
        else:
            self.p = p.T

        if alpha is None:
            alpha = 0.95
        self.c = matrix(np.hstack((np.zeros(self.I), [1], [1 / (1 - alpha)])))

        self._set_options(kwargs.get('options', globals()['cvar_options']))
        self.mean = self.p @ R
        self._expected_return_row = matrix(np.hstack((
            -self.mean_scalar * self.mean, np.zeros((1, 2)))))
        if self.demean:
            self.losses = -self.R_scalar * (R - self.mean)
        else:
            self.losses = -self.R_scalar * R

    def _set_options(self, options):
        self.demean = options.get('demean', True)
        if type(self.demean) != bool:
            raise ValueError('demean must be a boolean equal to True or False.')
        self.R_scalar = options.get('R_scalar', 1000)
        if type(self.R_scalar) not in (int, float) or self.R_scalar <= 0:
            raise ValueError('R_scalar must be a postive integer or float.')
        self.mean_scalar = options.get('mean_scalar', 100)
        if type(self.mean_scalar) not in (int, float) or self.mean_scalar <= 0:
            raise ValueError('mean_scalar must be a postive integer or float.')
        self.maxiter = options.get('maxiter', 500)
        if type(self.maxiter) != int or self.maxiter < 100:
            raise ValueError('maxiter must be a postive integer greater than or equal to 100.')
        self.reltol = options.get('reltol', 1e-8)
        if not 1e-8 <= self.reltol <= 1e-4:
            raise ValueError('reltol must be in [1e-8, 1e-4].')
        self.abstol = options.get('abstol', 1e-8)
        if not 1e-8 <= self.abstol <= 1e-4:
            raise ValueError('abstol must be in [1e-8, 1e-4].')

    def _benders_algorithm(self, G: sparse, h: matrix) -> np.ndarray:
        """Method for running Benders algorithm.

        Args:
            G: Inequality constraints matrix with shape (N, I) or (N+1, I).
            h: Inequality constraints vector with shape (N, 1) or (N+1, I).

        Returns:
            Solution to the mean-CVaR minimization problem.
        """
        eta = self.p @ self.losses
        p = 1
        solution, w, F_lower, G_benders, h_benders, eta, p = self._benders_main(G, h, eta, p)
        F_star = F_lower + self.c[-1] * (w - solution[-1])
        v = 1
        while self._benders_stopping_criteria(F_star, F_lower) and v <= self.maxiter:
            solution, w, F_lower, G_benders, h_benders, eta, p = self._benders_main(
                G_benders, h_benders, eta, p)
            F_star = min(F_lower + self.c[-1] * (w - solution[-1]), F_star)
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
            solvers.lp(c=self.c, G=G_benders, h=h_benders, A=self.A, b=self.b, solver='glpk')['x'])
        eta, p = self._benders_cut(solution)
        w = eta @ solution[0:-2] - p * solution[-2]
        F_lower = self.c.T @ solution
        return solution, w, F_lower, G_benders, h_benders, eta, p

    def _benders_cut(self, solution: np.ndarray) -> Tuple[np.ndarray, float]:
        """Method for generating benders cut variables.

        Args:
            solution: Current solution.

        Returns:
            Variables for the next cut.
        """
        K = (self.losses @ solution[0:self.I] >= solution[-2])[:, 0]
        eta = self.p[:, K] @ self.losses[K, :]
        p = np.sum(self.p[0, K])
        return eta, p

    def _benders_stopping_criteria(self, F_star: float, F_lower: float) -> bool:
        """Method for assessing if the algorithm should continue.

        Args:
            F_star: Best upper bound on the objective value.
            F_lower: Current lower bound on the objective value.

        Returns:
            Bolean indicating whether the algorithm should continue.
        """
        F_lower_abs = np.abs(F_lower)
        if F_lower_abs > 1e-10 and (F_star - F_lower) / F_lower_abs > self.reltol:
            return True
        elif F_lower_abs <= 1e-10 and (F_star - F_lower) > self.abstol:
            return True
        else:
            return False

    def efficient_portfolio(self, return_target: float = None) -> np.ndarray:
        """Method for computing an efficient portfolio with return target.

        Args:
            return_target: Return target for the efficient portfolio. If given
                None, the minimum CVaR portfolio is estimated.

        Returns:
            Efficient portfolio exposures with shape (I, 1).
        """
        if return_target is None:
            G = copy(self.G)
            h = copy(self.h)
        else:
            G = sparse([self.G, self._expected_return_row])
            h = matrix([self.h, -self.mean_scalar * return_target])
        solution = self._benders_algorithm(G, h)
        return solution[0:-2]

    def _calculate_max_expected_return(self) -> float:
        """Method for calculating the highest expected return and checking feasibility/boundness.

        Returns:
            Highest expected return for the given constraints.

        Raises:
            ValueError: If constraints are infeasible or _max_expected_return unbounded.
        """
        solution = solvers.lp(
            c=self._expected_return_row.T, G=self.G, h=self.h, A=self.A, b=self.b, solver='glpk')
        if solution['status'] == 'optimal':
            return -solution['primal objective'] / self.mean_scalar
        else:
            raise ValueError('Constraints are infeasible or _max_expected_return is unbounded.')

    def efficient_frontier(self, num_portfolios: int = None) -> np.ndarray:
        """Method for computing the efficient frontier.

        Args:
            num_portfolios: Number of portfolios used to span the efficient frontier. Default: 9.

        Returns:
            Efficient frontier with shape (I, num_portfolios).

        Raises:
            ValueError: If constraints are infeasible or _max_expected_return unbounded.
        """
        if num_portfolios is None:
            num_portfolios = 9

        _max_expected_return = self._calculate_max_expected_return()
        frontier = np.full((self.I, num_portfolios), np.nan)
        frontier[:, 0] = self.efficient_portfolio()[:, 0]
        _min_expected_return = float(self.mean @ frontier[:, 0])
        _delta = (_max_expected_return - _min_expected_return) / (num_portfolios - 1)

        for p in range(1, num_portfolios):
            frontier[:, p] = self.efficient_portfolio(_min_expected_return + _delta * p)[:, 0]

        return frontier
