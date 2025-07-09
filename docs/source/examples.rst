Examples
========

Jupyter Notebook examples are available in the `fortitudo.tech GitHub repository
<https://github.com/fortitudo-tech/fortitudo.tech/tree/main/examples>`_.
The repository contains the following examples:

1) How to combine CVaR optimization with Entropy Pooling views / stress-tests
2) A replication of the results from :cite:t:`Vorobets2021` for the original
   Entropy Pooling heuristic and a separate bonus sequential Entropy Pooling
   example using the H1 heuristic to implement views on S&P 500 and STOXX 50
   mean and volatility
3) The accompanied code for :cite:t:`Vorobets2022` with a comparison of
   mean-CVaR and mean-variance optimization explaining why we use demeaned
   CVaR as default (it's not recommended to run this example with 1,000,000
   scenarios on Binder, see the comments below)
4) An illustration of how to work with the time series simulation that follows
   with this package
5) The accompanied code for :cite:t:`Vorobets2022a` with an example of how to
   use the relative market values :math:`v` parameter for portfolio optimization
6) How to use the exponential decay simulation / P&L modeling functionality with
   historical time series for FAANG stocks
7) How to use the time series simulation for risk factor and P&L simulation and
   combine this with Entropy Pooling views on risk factors
8) The accompanied code for :cite:t:`Vorobets2023` illustrating how Bayesian
   networks can be used in combination with Entropy Pooling for causal and
   predictive market views and stress-testing
9) The accompanied code for :cite:t:`KristensenVorobets2024`, illustrating
   the effect of parameter uncertainty and introducing Exposure Stacking.
10) The accompanied code for :cite:t:`Vorobets2024`, illustrating how to
    optimize derivative portfolios using Entropy Pooling and Expsoure Stacking
11) The accompanied code for :cite:t:`KristensenVorobets2025`, illustrating
    the Fully Flexible Resampling method introduced in the
    `Portfolio Construction and Risk Management Book <https://antonvorobets.substack.com/p/pcrm-book>`_
12) The accompanied code for :cite:t:`Vorobets2025` that performs tests
    for normality of US equity index returns and rejects the Aggregational
    Gaussianity hypothesis

Watch this `YouTube playlist <https://www.youtube.com/playlist?list=PLfI2BKNVj_b2rurUsCtc2F8lqtPWqcs2K>`_
for a walkthrough of the package's functionality and examples. The examples are
good places to start exploring the functionality of this package.

For a high-level introduction to the investment framework, see this `YouTube video <https://youtu.be/4ESigySdGf8>`_
and `Substack post <https://open.substack.com/pub/antonvorobets/p/entropy-pooling-and-cvar-portfolio-optimization-in-python-ffed736a8347>`_.

For a mathematical introduction to the investment framework, see these
`SSRN articles <https://ssrn.com/author=2738420>`_.

For a pedagogical and deep presentation of the investment framework, see the
`Portfolio Construction and Risk Management Book <https://antonvorobets.substack.com/p/pcrm-book>`_.

To build the deepest understanding of all the theories and methods, you can
complete the `Applied Quantitative Investment Management Course <https://antonvorobets.substack.com/t/course>`_.

You can explore the examples in the cloud without any local installations using
`Binder <https://mybinder.org/v2/gh/fortitudo-tech/fortitudo.tech/main?labpath=examples>`_.
However, note that Binder servers have very limited resources and might not support
some of the optimized routines this package uses. If you want access to a stable
and optimized environment with persistent storage, please subscribe to our Data
Science Server.
