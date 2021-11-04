Examples
========

This page contains two examples:

1) How to combine Entropy Pooling views / stress-tests with CVaR optimization,
2) A replication of the results from :cite:t:`SeqEntropyPooling` for the original
   Entropy Pooling method.

The data used in all of the example is the simulation from :cite:t:`SeqEntropyPooling`.

General Overview
----------------

This example walks through the mean-CVaR and Entropy Pooling functionality
and illustrates how these two technologies can be combined.

We first load the necessary packages, data, and print some P&L stats:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import fortitudo.tech as ft

    R = ft.load_pnl()
    instrument_names = list(R.columns)
    means = np.mean(R, axis=0)
    vols = np.std(R, axis=0)
    stats_prior = pd.DataFrame(
        data=np.vstack((means, vols)).T,
        index=instrument_names,
        columns=['Mean', 'Volatility'])
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

The next step is to input the P&L, constraints, and probability vector into a
MeanCVaR object as well as optimize portfolios:

.. code-block:: python

    ft.cvar_options['demean'] = False
    R = R.values
    cvar_opt = ft.MeanCVaR(R, G=G_pf, h=h_pf, p=p)
    w_min = cvar_opt.efficient_portfolio()
    w_target = cvar_opt.efficient_portfolio(return_target=0.05)

Note that the MeanCVaR object uses demeaned P&L by default when optimizing the
portfolio's CVaR, as we believe it is best not to rely on the expected return
estimates in both portfolio risk and portfolio mean. In the above example, we
illustrate how you can disable this feature and specify that the optimization
should compute portfolio CVaR including its expected return.

Let us now assume that we have performed some analysis and concluded that the
mean of Private Equity should be 10%, while its volatility should be greater
than or equal to 33%. Entropy Pooling allows us to incorporate this market view
into our P&L assumption in a way that introduces the least amount of spurious
structure, measured by the relative entropy between our prior and posterior
probability vectors.

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
        data=np.vstack((means_post, vols_post)).T,
        index=instrument_names,
        columns=['Mean', 'Volatility'])
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
incorporates views / stress-tests in a way that respects the dependencies
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
        data=np.hstack((w_min, w_min_post)),
        index=instrument_names,
        columns=['Prior', 'Posterior'])
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
        data=np.hstack((w_target, w_target_post)),
        index=instrument_names,
        columns=['Prior', 'Posterior'])
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

We can also compute efficient frontiers for the prior and posterior probabilities:

.. code-block:: python 

    front = cvar_opt.efficient_frontier()
    print(np.round(pd.DataFrame(front * 100, index=instrument_names), 1))
    front_post = cvar_opt_post.efficient_frontier()
    print(np.round(pd.DataFrame(front_post * 100, index=instrument_names), 1))

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

From the allocation results, we note that the portfolios suffer from the 
well-known issues of concentrated portfolios. There are several ways of 
addressing this issue in practice, e.g., take parameter uncertainty into
account and introduce transaction costs or turnover constraints with an
initially diversified portfolio. These topics are however beyond the
scope of this example and package.

Entropy Pooling
---------------

This example replicates Table 4 and Table 7 in :cite:t:`SeqEntropyPooling`.
You can `download the article using this link <https://ssrn.com/abstract_id=3936392>`_
and compare the results.

We first load the necessary packages and P&L data as well as create a prior
probability vector :math:`p`:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import fortitudo.tech as ft

    R = ft.load_pnl()
    instrument_names = R.columns
    R = R.values
    S, I = R.shape
    p = np.ones((S, 1)) / S

Next, we compute and print some prior stats:

.. code-block:: python

    means_prior = p.T @ R
    vols_prior = np.sqrt(p.T @ (R - means_prior)**2)
    skews_prior = p.T @ ((R - means_prior) / vols_prior)**3
    kurts_prior = p.T @ ((R - means_prior) / vols_prior)**4
    corr_prior = np.corrcoef(R.T)

    data_prior = np.hstack((
        np.round(means_prior.T * 100, 1), np.round(vols_prior.T * 100, 1),
        np.round(skews_prior.T, 2), np.round(kurts_prior.T, 2)))
    prior_df = pd.DataFrame(
        data=data_prior,
        index=instrument_names,
        columns=['Mean', 'Volatility', 'Skewness', 'Kurtosis'])
    print(prior_df)

    corr_prior_df = pd.DataFrame(
        data=np.intc(np.round(corr_prior * 100)),
        index=enumerate(instrument_names, start=1),
        columns=range(1, I + 1))
    print(corr_prior_df)

This gives the following output (Table 1 and Table 5 in
:cite:t:`SeqEntropyPooling`)::

                    Mean  Volatility  Skewness  Kurtosis
    Gov & MBS       -0.7         3.2      0.10      3.02
    Corp IG         -0.4         3.4      0.11      3.11
    Corp HY          1.9         6.1      0.17      2.97
    EM Debt          2.7         7.5      0.22      3.06
    DM Equity        6.4        14.9      0.40      3.15
    EM Equity        8.0        26.9      0.77      4.10
    Private Equity  13.7        27.8      0.72      3.76
    Infrastructure   5.9        10.8      0.31      3.19
    Real Estate      4.3         8.1      0.23      3.09
    Hedge Funds      4.8         7.2      0.20      3.05

                          1    2    3    4    5    6    7    8    9    10
    (1, Gov & MBS)       100   60    0   30  -20  -10  -30  -10  -20  -20
    (2, Corp IG)          60  100   50   60   10   20   10   10   10   30
    (3, Corp HY)           0   50  100   60   60   69   59   30   30   70
    (4, EM Debt)          30   60   60  100   40   59   30   20   20   40
    (5, DM Equity)       -20   10   60   40  100   69   79   40   40   80
    (6, EM Equity)       -10   20   69   59   69  100   69   30   39   79
    (7, Private Equity)  -30   10   59   30   79   69  100   39   49   79
    (8, Infrastructure)  -10   10   30   20   40   30   39  100   40   40
    (9, Real Estate)     -20   10   30   20   40   39   49   40  100   50
    (10, Hedge Funds)    -20   30   70   40   80   79   79   40   50  100

We then specify the same views as the article: mean of Private Equity is 10%,
volatility of EM Equity is less than or equal to 20%, skewness of DM Equity is
less than or equal to −0.75, kurtosis of DM Equity is greater than or equal to
3.5, and correlation between Corp HY and EM Debt is 50%.

.. code-block:: python

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

Next, we calculate the posterior probabilities :math:`q`, relative entropy (RE),
and effective number of scenarios (ENS).

.. code-block:: python

    q = ft.entropy_pooling(p, A, b, G, h)
    relative_entropy = q.T @ (np.log(q) - np.log(p))
    effective_number_scenarios = np.exp(-relative_entropy)

Means, volatilities, skewness, kurtosis, and the correlation matrix are then
recalculated using the posterior probabilities.

.. code-block:: python

    means_post = q.T @ R
    vols_post = np.sqrt(q.T @ (R - means_post)**2)
    skews_post = q.T @ ((R - means_post) / vols_post)**3
    kurts_post = q.T @ ((R - means_post) / vols_post)**4
    cov_post = np.cov(R, rowvar=False, aweights=q[:, 0])
    vols_inverse = np.diag(vols_post[0, :]**-1)
    corr_post = vols_inverse @ cov_post @ vols_inverse

Finally, we print the posterior results:

.. code-block:: python

    data_post = np.hstack((
        np.round(means_post.T * 100, 1), np.round(vols_post.T * 100, 1),
        np.round(skews_post.T, 2), np.round(kurts_post.T, 2)))
    post_df = pd.DataFrame(
        data=data_post,
        index=instrument_names,
        columns=['Mean', 'Volatility', 'Skewness', 'Kurtosis'])
    print(post_df)

    print(f'ENS = {np.round(effective_number_scenarios[0, 0] * 100, 2)}%.')
    print(f'RE = {np.round(relative_entropy[0, 0] * 100, 2)}%.')

    corr_post_df = pd.DataFrame(
        data=np.intc(np.round(corr_post * 100)),
        index=enumerate(instrument_names, start=1),
        columns=range(1, I + 1))
    print(corr_post_df)

Which gives the following output (Table 4 and Table 7 in 
:cite:t:`SeqEntropyPooling`)::

                    Mean  Volatility  Skewness  Kurtosis
    Gov & MBS       -0.6         3.2      0.06      2.91
    Corp IG         -0.5         3.4      0.14      3.12
    Corp HY          1.9         6.1     -0.06      2.97
    EM Debt          2.7         7.5      0.13      3.07
    DM Equity        6.4        14.9     -0.75      3.50
    EM Equity        8.0        20.0     -0.22      3.34
    Private Equity  10.0        24.3      0.12      3.17
    Infrastructure   5.7        10.6      0.28      3.16
    Real Estate      3.7         8.0      0.13      3.02
    Hedge Funds      4.6         7.0     -0.62      3.81

    ENS = 70.92%.
    RE = 34.36%.

                          1    2    3    4    5    6    7    8    9    10
    (1, Gov & MBS)       100   60   -2   35  -23  -10  -34  -10  -20  -24
    (2, Corp IG)          60  100   51   63    9   20    7    9   11   29
    (3, Corp HY)          -2   51  100   50   57   64   55   27   27   67
    (4, EM Debt)          35   63   50  100   31   51   16   16   15   29
    (5, DM Equity)       -23    9   57   31  100   66   76   37   38   79
    (6, EM Equity)       -10   20   64   51   66  100   62   27   36   75
    (7, Private Equity)  -34    7   55   16   76   62  100   38   47   76
    (8, Infrastructure)  -10    9   27   16   37   27   38  100   39   38
    (9, Real Estate)     -20   11   27   15   38   36   47   39  100   49
    (10, Hedge Funds)    -24   29   67   29   79   75   76   38   49  100

The results for the sequential heuristics are not replicated as they are a
part of Fortitudo Technologies' proprietary software, which also contains an
elegant interface for handling the different views instead of manually specifying
them through :math:`A`, :math:`b`, :math:`G`, and :math:`h`. The interested
reader can replicate the results of the sequential heuristics by using the P&L
simulation that follows with this package and the Entropy Pooling technology.