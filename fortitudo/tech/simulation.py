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
import pandas as pd
from typing import Union, Tuple
from .functions import covariance_matrix


def exp_decay_probs(R: Union[pd.DataFrame, np.ndarray], half_life: int) -> np.ndarray:
    """Function for computing exponential decay probabilities.

    Args:
        R: P&L / risk factor simulation with shape (T, I).
        half_life: Exponential decay half life.

    Returns:
        Exponentially decaying probabilities vector with shape (T, 1).
    """
    T = R.shape[0]
    p = np.exp(-np.log(2) / half_life * (T - np.arange(1, T + 1)))
    return (p / np.sum(p))[:, np.newaxis]


def normal_exp_decay_calib(
        R: Union[pd.DataFrame, np.ndarray], half_life: int
        ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.DataFrame, pd.DataFrame]]:
    """Function for computing exponential decay mean vector and covariance matrix.

    Args:
        R: P&L / risk factor simulation with shape (T, I).
        half_life: Exponential decay half life.

    Returns:
        Mean vector with shape (I, 1) and covariance matrix with shape (I, I).
    """
    p = exp_decay_probs(R, half_life)
    mean = R.T @ p
    cov_matrix = covariance_matrix(R, p)

    if type(mean) is pd.DataFrame:
        mean.columns = ['Mean']
    elif type(mean) is np.ndarray:
        cov_matrix = cov_matrix.values

    return mean, cov_matrix
