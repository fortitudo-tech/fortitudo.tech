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

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fortitudo.tech import (
    entropy_pooling, MeanCVaR, cvar_options, MeanVariance, load_parameters,
    simulation_moments, covariance_matrix, correlation_matrix, portfolio_cvar,
    portfolio_var, portfolio_vol, load_pnl, load_risk_factors, load_time_series,
    plot_vol_surface, forward, call_option, put_option, FullyFlexibleResampling,
    exp_decay_probs, normal_exp_decay_calib, exposure_stacking)

from fortitudo.tech.functions import _simulation_check

R = load_pnl()
time_series = load_time_series()
