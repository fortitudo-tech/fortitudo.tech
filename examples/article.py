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

"""
This file replicates the results (Table 4 and Table 7) for the original
discrete EP method from the Sequential Entropy Pooling Heuristics article,
available on https://ssrn.com/abstract=3936392.
"""

import numpy as np
import fortitudo.tech as ft

# Load P&L
R = ft.load_pnl()
instrument_names = R.columns
R = R.values
S, I = R.shape
p = np.ones((S, 1)) / S

# Compute prior stats
means_prior = p.T @ R
vols_prior = np.sqrt(p.T @ (R - means_prior)**2)
skews_prior = p.T @ ((R - means_prior) / vols_prior)**3
kurts_prior = p.T @ ((R - means_prior) / vols_prior)**4
corr_prior = np.corrcoef(R.T)

# Create views matrices and vectors
mean_rows = R[:, 2:7].T
vol_rows = (R[:, 2:6] - means_prior[:, 2:6]).T**2
skew_row = ((R[:, 4] - means_prior[:, 4]) / vols_prior[:, 4])**3
kurt_row = ((R[:, 4] - means_prior[:, 4]) / vols_prior[:, 4])**4
corr_row = (R[:, 2] - means_prior[:, 2]) * (R[:, 3] - means_prior[:, 3])
A = np.vstack((np.ones((1, S)), mean_rows, vol_rows[0:-1, :], corr_row[np.newaxis, :]))
b = np.vstack(([1], means_prior[:, 2:6].T, [0.1], vols_prior[:, 2:5].T**2,
               [0.5 * vols_prior[0, 2] * vols_prior[0, 3]]))
G = np.vstack((vol_rows[-1, :], skew_row, -kurt_row))
h = np.array([[0.2**2], [-0.75], [-3.5]])

# Compute posterior probabilities, re and ens
q = ft.entropy_pooling(p, A, b, G, h)
relative_entropy = q.T @ (np.log(q) - np.log(p))
effective_number_scenarios = np.exp(-relative_entropy)

# Compute posterior stats
means_post = q.T @ R
vols_post = np.sqrt(q.T @ (R - means_post)**2)
skews_post = q.T @ ((R - means_post) / vols_post)**3
kurts_post = q.T @ ((R - means_post) / vols_post)**4
cov_post = np.zeros((I, I))
for s in range(S):
    cov_post += q[s, 0] * (R[s, :] - means_post).T @ (R[s, :] - means_post)
vols_inverse = np.diag(vols_post[0, :]**-1)
corr_post = vols_inverse @ cov_post @ vols_inverse
