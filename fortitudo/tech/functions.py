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
import pandas as pd
from typing import Tuple, Union


def _simulation_check(
        R: Union[pd.DataFrame, np.ndarray], p: np.ndarray) -> Tuple[list, np.ndarray, np.ndarray]:
    """Function for preprocessing simulation and probability vector input.

    Args:
        R: P&L / risk factor simulation with shape (S, I).
        p: probability vector with shape (S, 1) or None.

    Returns:
        Validated and preprocessed simulation_names, R, and p.
    """
    if type(R) == pd.DataFrame:
        simulation_names = R.columns
        R = R.values
    else:
        simulation_names = np.arange(R.shape[1])

    S = R.shape[0]
    if p is None:
        p = np.ones((S, 1)) / S
    elif p.shape[0] != S:
        raise ValueError('R and p must have the same length.')

    return simulation_names, R, p


def simulation_moments(R: Union[pd.DataFrame, np.ndarray], p: np.ndarray = None) -> pd.DataFrame:
    """Function for computing simulation moments (mean, volatility, skewness, and kurtosis).

    Args:
        R: P&L / risk factor simulation with shape (S, I).
        p: probability vector with shape (S, 1). Default np.ones((S, 1)) / S.

    Returns:
        DataFrame with shape (I, 4) containing simulation moments.
    """
    simulation_names, R, p = _simulation_check(R, p)
    means = p.T @ R
    R_demean = R - means
    vols = np.sqrt(p.T @ R_demean**2)
    R_standardized = R_demean / vols
    skews = p.T @ R_standardized**3
    kurts = p.T @ R_standardized**4
    results = pd.DataFrame(np.hstack((means.T, vols.T, skews.T, kurts.T)),
                           index=simulation_names,
                           columns=['Mean', 'Volatility', 'Skewness', 'Kurtosis'])
    return results


def covariance_matrix(R: Union[pd.DataFrame, np.ndarray], p: np.ndarray = None) -> pd.DataFrame:
    """Function for computing the covariance matrix.

    Args:
        R: P&L / risk factor simulation with shape (S, I).
        p: probability vector with shape (S, 1). Default np.ones((S, 1)) / S.

    Returns:
        Covariance matrix with shape (I, I).
    """
    simulation_names, R, p = _simulation_check(R, p)
    cov = np.cov(R, rowvar=False, aweights=p[:, 0])
    return pd.DataFrame(cov, index=enumerate(simulation_names))


def correlation_matrix(R: Union[pd.DataFrame, np.ndarray], p: np.ndarray = None) -> pd.DataFrame:
    """Function for computing the correlation matrix.

    Args:
        R: P&L / risk factor simulation with shape (S, I).
        p: probability vector with shape (S, 1). Default np.ones((S, 1)) / S.

    Returns:
        Correlation matrix with shape (I, I).
    """
    cov = covariance_matrix(R, p)
    vols_inverse = np.diag(np.sqrt(np.diag(cov.values))**-1)
    corr = vols_inverse @ cov.values @ vols_inverse
    return pd.DataFrame(corr, index=cov.index)
