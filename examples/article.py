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
import pandas as pd
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

# Print prior stats
data_prior = np.hstack((
    np.round(means_prior.T * 100, 1), np.round(vols_prior.T * 100, 1),
    np.round(skews_prior.T, 2), np.round(kurts_prior.T, 2)))
prior_df = pd.DataFrame(
    data_prior, index=instrument_names,
    columns=['Mean', 'Volatility', 'Skewness', 'Kurtosis'])
print(prior_df)

corr_prior_df = pd.DataFrame(
    np.intc(np.round(corr_prior * 100)),
    index=enumerate(instrument_names, start=1),
    columns=range(1, I + 1))
print(corr_prior_df)

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
cov_post = np.cov(R, rowvar=False, aweights=q[:, 0])
vols_inverse = np.diag(vols_post[0, :]**-1)
corr_post = vols_inverse @ cov_post @ vols_inverse

# Print posterior stats
data_post = np.hstack((
    np.round(means_post.T * 100, 1), np.round(vols_post.T * 100, 1),
    np.round(skews_post.T, 2), np.round(kurts_post.T, 2)))
post_df = pd.DataFrame(
    data_post, index=instrument_names,
    columns=['Mean', 'Volatility', 'Skewness', 'Kurtosis'])
print(post_df)

print(f'ENS = {np.round(effective_number_scenarios[0, 0] * 100, 2)}%.')
print(f'RE = {np.round(relative_entropy[0, 0] * 100, 2)}%.')

corr_post_df = pd.DataFrame(
    np.intc(np.round(corr_post * 100)),
    index=enumerate(instrument_names, start=1),
    columns=range(1, I + 1))
print(corr_post_df)
