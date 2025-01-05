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
from pandas import DataFrame
from matplotlib.figure import Figure
from context import load_parameters, load_risk_factors, time_series, plot_vol_surface, R


def test_load_data():
    instrument_names, means, covariance_matrix = load_parameters()
    assert list(R.columns) == instrument_names
    I = R.shape[1]
    assert means.shape == (I,)
    assert covariance_matrix.shape == (I, I)


def test_load_risk_factors():
    risk_factors = load_risk_factors()
    assert type(risk_factors) is DataFrame
    assert risk_factors.shape == (5039, 82)


def test_time_series():
    assert time_series.shape == (5040, 79)
    assert np.all(time_series.values >= 0)


def test_plot_vol_surface():
    fig, _ = plot_vol_surface(0, time_series.values[:, 34:69])
    assert type(fig) is Figure
