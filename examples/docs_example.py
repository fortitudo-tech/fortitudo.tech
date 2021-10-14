import numpy as np
import pandas as pd
import fortitudo.tech as ft

# Load data
R = pd.read_csv('pnl.csv')
instrument_names = list(R.columns)

# Compute prior stats
means = np.mean(R, axis=0)
vols = np.std(R, axis=0)
stats_prior = pd.DataFrame(
    np.vstack((means, vols)).T, index=instrument_names, columns=['Mean', 'Volatility'])
print(np.round(stats_prior * 100, 1))

# P&L and prior probability parameters
S, I = R.shape
p = np.ones((S, 1)) / S

# Portfolio constraints
G_pf = np.vstack((np.eye(I), -np.eye(I)))
h_pf = np.hstack((0.25 * np.ones(I), np.zeros(I)))

# CVaR optimization object
ft.cvar_options['demean'] = False
R = R.values
cvar_opt = ft.MeanCVaR(R, G=G_pf, h=h_pf, p=p)
w_min = cvar_opt.efficient_portfolio()
w_target = cvar_opt.efficient_portfolio(return_target=0.05)

# Stress-test P&L assumptions with Entropy Pooling
expected_return_row = R[:, 6][np.newaxis, :]
variance_row = (expected_return_row - 0.1) * (expected_return_row - 0.1)
A = np.vstack((np.ones((1, S)), expected_return_row))
b = np.array([[1], [0.1]])
G = -variance_row
h = np.array([[-0.33**2]])
q = ft.entropy_pooling(p, A, b, G, h)

# Compute posterior stats
means_post = q.T @ R
vols_post = np.sqrt(q.T @ (R - means_post)**2)
stats_post = pd.DataFrame(
    np.vstack((means_post, vols_post)).T, index=instrument_names, columns=['Mean', 'Volatility'])
print(np.round(stats_post * 100, 1))

# Optimize with posterior probabilities
cvar_opt_post = ft.MeanCVaR(R, G=G_pf, h=h_pf, p=q)
w_min_post = cvar_opt_post.efficient_portfolio()
w_target_post = cvar_opt_post.efficient_portfolio(return_target=0.05)

# Compare portfolios
min_risk_pfs = pd.DataFrame(
    np.hstack((w_min, w_min_post)), index=instrument_names, columns=['Prior', 'Posterior'])
print(np.round(min_risk_pfs * 100, 2))
target_return_pfs = pd.DataFrame(
    np.hstack((w_target, w_target_post)), index=instrument_names, columns=['Prior', 'Posterior'])
print(np.round(target_return_pfs * 100, 2))
