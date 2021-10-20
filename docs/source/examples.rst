Examples
--------

This example walks through the mean-CVaR and Entropy Pooling functionality
and illustrates how these two technologies can be combined.

We first load the necessary packages, data, and print some P&L stats:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import fortitudo.tech as ft

    R = pd.read_csv('pnl.csv')
    instrument_names = list(R.columns)
    means = np.mean(R, axis=0)
    vols = np.std(R, axis=0)
    stats_prior = pd.DataFrame(
        np.vstack((means, vols)).T, index=instrument_names, columns=['Mean', 'Volatility'])
    print(np.round(stats_prior * 100, 1))

This gives the following result::

                    Mean  Volatility
    Gov & MBS       -0.7         3.2
    Corp IG         -0.4         3.4
    Corp HY          1.9         6.1
    EM Debt          2.7         7.5
    DM Equity        6.4        14.9
    EM Equity        8.0        26.9
    Private Equity  13.7        27.8
    Infrastructure   5.9        10.8
    Real Estate      4.3         8.1
    Hedge Funds      4.8         7.2

Next, we extract P&L dimension parameters :math:`S` and :math:`I`, specify a prior
probability vector :math:`p`, and create some portfolio constraints:

.. code-block:: python

    S, I = R.shape
    p = np.ones((S, 1)) / S
    G_pf = np.vstack((np.eye(I), -np.eye(I)))
    h_pf = np.hstack((0.25 * np.ones(I), np.zeros(I)))

The above portfolio constraints simply specify that it is a long-only portfolio
with an upper bound of 25% for individual assets. This ensures that the optimized
portfolios are invested in at least 4 assets and imposes some diversification.

The next step is to input the P&L, constraints, and probability vector into the
MeanCVaR object as well as optimize portfolios:

.. code-block:: python

    ft.cvar_options['demean'] = False
    R = R.values
    cvar_opt = ft.MeanCVaR(R, G=G_pf, h=h_pf, p=p)
    w_min = cvar_opt.efficient_portfolio()
    w_target = cvar_opt.efficient_portfolio(return_target=0.05)

Note that the MeanCVaR object uses demeaned P&L by default when optimizing the
portfolio's CVaR, as we believe it is best not to rely on the expected return
estimates in both the risk and the expectation. In the above example, we
illustrate how you can disable this feature and specify that the optimization
should compute portfolio CVaR including its expected return.

Let us now assume that we have done some analysis and concluded that the mean
of Private Equity should be 10%, while its volatility should be greater than
or equal to 33%. Entropy Pooling allows us to incorporate this market view
into our P&L assumption in a way that introduces the least amount of spurious
structure, which is measured by the relative entropy between our prior and
posterior probability vectors.

The above views for Private Equity are implemented below:

.. code-block:: python

    expected_return_row = R[:, 6][np.newaxis, :]
    variance_row = (expected_return_row - 0.1)**2
    A = np.vstack((np.ones((1, S)), expected_return_row))
    b = np.array([[1], [0.1]])
    G = -variance_row
    h = np.array([[-0.33**2]])
    q = ft.entropy_pooling(p, A, b, G, h)

    means_post = q.T @ R
    vols_post = np.sqrt(q.T @ (R - means_post)**2)
    stats_post = pd.DataFrame(
        np.vstack((means_post, vols_post)).T, index=instrument_names, columns=['Mean', 'Volatility'])
    print(np.round(stats_post * 100, 1))

Which gives the following posterior means and volatilities::

                    Mean  Volatility
    Gov & MBS       -0.5         3.2
    Corp IG         -0.5         3.4
    Corp HY          1.2         6.4
    EM Debt          2.3         7.6
    DM Equity        4.4        16.4
    EM Equity        5.2        29.2
    Private Equity  10.0        33.0
    Infrastructure   5.1        11.1
    Real Estate      3.6         8.5
    Hedge Funds      3.8         8.0

We note that our views regarding Private Equity are satisfied. In addition, 
we note that volatilities of the riskier assets have increased, while their
expected returns have decreased. This illustrates how Entropy Pooling
incorporates views/stress-tests in a way that tries to respect the dependencies
of the prior distribution.

With the posterior probabilities at hand, we want to examine the effect of our
views on the efficient CVaR portfolios. This is easy to do by simply specifying
that the posterior probability vector :math:`q` should be used in the CVaR
optimization:

.. code-block:: python

    cvar_opt_post = ft.MeanCVaR(R, G=G_pf, h=h_pf, p=q)
    w_min_post = cvar_opt_post.efficient_portfolio()
    w_target_post = cvar_opt_post.efficient_portfolio(return_target=0.05)

We can then print the results of the optimization and compare allocations.
First for the minimum risk portfolios:

.. code-block:: python

    min_risk_pfs = pd.DataFrame(
    np.hstack((w_min, w_min_post)), index=instrument_names, columns=['Prior', 'Posterior'])
    print(np.round(min_risk_pfs * 100, 1))

Which gives the following output::

                    Prior  Posterior
    Gov & MBS        25.0       25.0
    Corp IG          25.0       25.0
    Corp HY           0.5        6.5
    EM Debt           3.9        5.0
    DM Equity         0.0        0.0
    EM Equity        -0.0        0.0
    Private Equity   -0.0        0.0
    Infrastructure    6.9        6.9
    Real Estate      14.5       17.7
    Hedge Funds      24.2       14.0

And then for the portfolios with an expected return target of 5%:

.. code-block:: python

    target_return_pfs = pd.DataFrame(
    np.hstack((w_target, w_target_post)), index=instrument_names, columns=['Prior', 'Posterior'])
    print(np.round(target_return_pfs * 100, 1))

Which gives the following output::

                    Prior  Posterior
    Gov & MBS         0.0       -0.0
    Corp IG           0.0        0.0
    Corp HY           0.0        0.0
    EM Debt          19.8        8.1
    DM Equity         0.0        0.0
    EM Equity         0.0        0.0
    Private Equity    5.2       16.9
    Infrastructure   25.0       25.0
    Real Estate      25.0       25.0
    Hedge Funds      25.0       25.0

It should be straightforward to make sense of these results. In the minimum
risk case, we see that we allocate less to the riskier assets that now have a
higher risk due to the higher volatility view. In the 5% target return case,
we note that we must allocate more to the riskier assets in order to reach
the 5% expected return target.

From the allocation results, we note that the portfolios suffer from the 
well-known issues of concentrated portfolios. There are several ways of 
addressing this issue in practice, e.g., take parameter uncertainty into
account and introduce transaction costs or turnover constraints with an
initially diversified portfolio. These topics are however beyond the
scope of this example and package.

We can also compute an efficient frontier for the prior probabilies. The number
of portfolios used to span the frontier is by default set to 9:

.. code-block:: python 

    front = cvar_opt.efficient_frontier()
    print(np.round(pd.DataFrame(front * 100, index=instrument_names), 1))


The gives the following output::

                    0     1     2     3     4     5     6     7     8
    Gov & MBS       25.0  25.0  18.8   5.6   0.0   0.0   0.0  -0.0   0.0
    Corp IG         25.0   8.3   0.0   0.0   0.0   0.0   0.0   0.0   0.0
    Corp HY          0.5   0.0   0.0   0.0   0.0   0.0  -0.0   0.0   0.0
    EM Debt          3.9   9.6  12.8  18.1  17.4  12.7   7.1  -0.0   0.0
    DM Equity        0.0   0.0   0.0   0.0   0.0   0.0   0.0  20.9  25.0
    EM Equity       -0.0   0.0   0.0  -0.0  -0.0   0.0   0.0   2.2  25.0
    Private Equity  -0.0  -0.0   0.7   2.2   7.6  15.4  23.1  25.0  25.0
    Infrastructure   6.9  12.9  18.9  24.1  25.0  25.0  25.0  25.0  25.0
    Real Estate     14.5  19.2  23.9  25.0  25.0  21.9  19.9   2.0   0.0
    Hedge Funds     24.2  25.0  25.0  25.0  25.0  25.0  25.0  25.0   0.0


The efficient frontier for the posterior probabilies is calculated:

.. code-block:: python

    front_post = cvar_opt_post.efficient_frontier()
    print(np.round(pd.DataFrame(front_post * 100, index=instrument_names), 1))

The gives the following output::

                       0     1     2     3     4     5     6     7     8
    Gov & MBS       25.0  25.0  25.0  14.1   0.0  -0.0  -0.0   0.0  -0.0
    Corp IG         25.0  16.0   0.8   0.0   0.0   0.0   0.0   0.0   0.0
    Corp HY          6.5   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
    EM Debt          5.0   6.9  13.5  14.3  23.3  15.6   8.0   0.3   0.0
    DM Equity        0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  25.0
    EM Equity        0.0  -0.0  -0.0   0.0   0.0   0.0   0.0   0.0  25.0
    Private Equity   0.0   0.0  -0.0  -0.0   1.7   9.4  17.0  24.7  25.0
    Infrastructure   6.9  12.4  15.9  23.8  25.0  25.0  25.0  25.0  25.0
    Real Estate     17.7  17.0  19.7  22.8  25.0  25.0  25.0  25.0  -0.0
    Hedge Funds     14.0  22.6  25.0  25.0  25.0  25.0  25.0  25.0   0.0


