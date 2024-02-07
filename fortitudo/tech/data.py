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
from matplotlib import cm
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from pkgutil import get_data
from io import StringIO
from typing import Tuple


def load_pnl() -> pd.DataFrame:
    """Function for loading the P&L simulation from https://ssrn.com/abstract=3936392,
    https://ssrn.com/abstract=4217884, and https://ssrn.com/abstract=4444291.

    Returns:
        P&L simulation.
    """
    pnl_string = StringIO(get_data('fortitudo.tech', 'data/pnl.csv').decode())
    return pd.read_csv(pnl_string)


def load_parameters() -> Tuple[list, np.ndarray, np.ndarray]:
    """Function for loading the P&L parameters from https://ssrn.com/abstract=4034316.

    Returns:
        Instrument names, means vector, and covariance matrix.
    """
    parameters_string = StringIO(get_data('fortitudo.tech', 'data/parameters.csv').decode())
    data = pd.read_csv(parameters_string)
    instrument_names = list(data.columns)
    data = data.values
    means = data[0, :]
    vols = np.diag(data[1, :])
    correlation_matrix = data[2:, :]
    covariance_matrix = vols @ correlation_matrix @ vols
    return instrument_names, means, covariance_matrix


def load_risk_factors() -> pd.DataFrame:
    """Function for loading the risk factor simulation from 7_RiskFactorViews.ipynb that is used
    in https://ssrn.com/abstract=4444291.

    Returns:
        Risk factor simulation.
    """
    rf_string = StringIO(get_data('fortitudo.tech', 'data/risk_factors.csv').decode())
    return pd.read_csv(rf_string)


def load_time_series() -> pd.DataFrame:
    """Function for loading an SDE based time series simulation.

    Returns:
        Time series simulation.
    """
    ts_string = StringIO(get_data('fortitudo.tech', 'data/time_series.csv').decode())
    return pd.read_csv(ts_string)


def plot_vol_surface(
        index: int, vol_surface: np.ndarray, figsize: Tuple[float, float] = None
        ) -> Tuple[Figure, Axes]:
    """Function for plotting the implied vol surface from the time series simulation.

    Args:
        index: Index for the implied vol surface scenario.
        vol_surface: Matrix with shape (T, 35) or (S, 35) containing the implied vols.
        figsize: Figure size. Default (10, 7).

    Returns:
        3d implied vol surface plot.
    """
    if figsize is None:
        figsize = (10, 7)

    strikes = [90, 95, 97.5, 100, 102.5, 105, 110]
    maturities = [1 / 12, 1 / 4, 1 / 2, 1, 2]
    strikes, maturities = np.meshgrid(strikes, maturities)
    vol_surface = np.reshape(vol_surface[index], (5, 7))
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize=figsize)
    ax.set_xlabel('Strike')
    ax.set_ylabel('Maturity')
    ax.set_zlabel('Implied vol')
    ax.plot_surface(strikes, maturities, vol_surface, cmap=cm.coolwarm)
    return fig, ax
