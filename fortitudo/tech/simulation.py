# fortitudo.tech - Novel Investment Technologies.
# Copyright (C) 2021-2025 Fortitudo Technologies.

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
from .entropy_pooling import entropy_pooling


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


class FullyFlexibleResampling:
    """Fully Flexible Resampling (FFR) class for simulaton with one state variable.

    The Fully Flexible Resampling (FFR) method was first introduced in Chapter 3 of
    the Portfolio Construction and Risk Management book.

    It is an instance of the more general Time and State-Dependent Resampling
    class documented in Kristensen and Vorobets (2025).

    Args:
        stationary_transformations: data containing the stationary transformations
            with shape (T_tilde, N_tilde).
    """
    def __init__(self, stationary_transformations: Union[pd.DataFrame, np.ndarray]):
        if len(stationary_transformations.shape) != 2:
            raise ValueError(
                'stationary_transformations must be a 2d matrix, '
                + f'given {stationary_transformations.shape}.')
        self._T, self._N = self._stationary_transformations.shape
        if type(stationary_transformations) is pd.DataFrame:
            self._stationary_transformations = stationary_transformations.values
        else:
            self._stationary_transformations = stationary_transformations

    def _compute_crisp_indices(
            self, state_variable: np.ndarray, conditioning_values: list[float],
            state_num: int) -> np.ndarray:
        conditioning_values.sort()
        crisp_indices = np.full((self._T, len(conditioning_values) + 1), np.nan)
        crisp_indices[:, 0] = (state_variable <= conditioning_values[0])
        if np.all(crisp_indices[:, 0] == 0):
            raise ValueError(
                f'State variable {state_num} has conditioning range (-infinity, '
                + f'{conditioning_values[0]}] which does not contain any observations.')
        for i, value in enumerate(conditioning_values[1:], start=1):
            indices = ((state_variable <= value) - np.sum(crisp_indices[:, 0:i], axis=1))
            if np.all(indices == 0):
                raise ValueError(
                    f'State variable {state_num} has conditioning range '
                    + f'[{conditioning_values[i - 1]}, {value}] which does not contain any '
                    + 'observations.')
            crisp_indices[:, i] = indices
        crisp_indices[:, -1] = (state_variable > conditioning_values[-1])
        if np.all(crisp_indices[:, -1] == 0):
            raise ValueError(
                f'State variable {state_num} has conditioning range [{conditioning_values[-1]}, '
                + 'infinity) which does not contain any observations.')
        return crisp_indices.astype(bool)

    def _entropy_pooling_state(
            self, p: np.ndarray, state: np.ndarray, mean: float, vol: float) -> np.ndarray:
        A = np.vstack((np.ones((1, self._T)), state.T))
        b = np.array([[1], [mean]])
        G = (state - mean).T**2
        h = np.array([[vol**2]])
        return entropy_pooling(p, A, b, G, h)

    def _individual_probabilities(
            self, p: np.ndarray, state_variable: np.ndarray, crisp_indices: np.ndarray) -> list:
        """Computes the posterior probabilities for each of the ranges."""
        individual_probabilities = []
        for i in range(crisp_indices.shape[1]):
            ind = crisp_indices[:, i]
            mean = np.mean(state_variable[ind])
            vol = np.std(state_variable[ind])
            individual_probabilities.append(
                self._entropy_pooling_state(p, state_variable, mean, vol))
        return individual_probabilities

    def compute_probabilities(
            self, state_variable, conditioning_values, half_life=None
            ) -> Tuple[np.ndarray, np.ndarray]:
        """Method for computing the Fully Flexible Resampling probabilities.

        Args:
            state_variable: time series for the state variable with shape (T_tilde, 1).
            conditioning_values: conditioning values used to define the crisp probability bands.
            half_life: half life parameter for exponentially decaying prior probabilities.

        Returns:
            Matrix with Fully Flexible Resampling probabilities and historical
            states vector.
        """

        return probabilities, state_vector

    def simulate(
            self, S, H, probabilities, state_vector, initial_state=None) -> np.ndarray:
        """Simulation method for Fully Flexible Resampling.

        Args:
            S: number of simulated future paths.
            H: simulation horizon.
            probabilities: the resampling probabilities for each state.
            state_vector: vector containing the historical states.
            initial_state: optional initial state. If given none, the latest
                state is used by default.

        Returns:
            Resampled stationary transformations simulations with shape (S, I, H).
        """

        return simulation
