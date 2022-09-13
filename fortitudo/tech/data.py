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

from pkgutil import get_data
from io import StringIO
from typing import Tuple
import numpy as np
import pandas as pd


def load_pnl() -> pd.DataFrame:
    """Function for loading the P&L simulation from https://ssrn.com/abstract=3936392
    and https://ssrn.com/abstract=4217884.

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


def load_time_series() -> pd.DataFrame:
    """Function for loading the time series simulation.

    Returns:
        Time series simulation.
    """
    ts_string = StringIO(get_data('fortitudo.tech', 'data/time_series.csv').decode())
    return pd.read_csv(ts_string)
