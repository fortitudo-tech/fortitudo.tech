Examples
========

Jupyter Notebook examples are available in the `fortitudo.tech GitHub repository
<https://github.com/fortitudo-tech/fortitudo.tech/tree/main/examples>`_.
The repository contains the following examples:

1) How to combine CVaR optimization with Entropy Pooling views / stress-tests
2) A replication of the results from :cite:t:`Vorobets2021` for the original
   Entropy Pooling heuristic
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
9) Example illustrating the effect of parameter uncertainty in mean-variance
   optimization.

The examples are good places to start exploring the functionality of this package.
We have very limited resources for support in relation to these, but please let
us know if you have suggestions for how we can improve them and make them easier
to understand.

You can explore the examples in the cloud without any local installations using
`Binder <https://mybinder.org/v2/gh/fortitudo-tech/fortitudo.tech/main?labpath=examples>`_.
However, note that Binder servers have very limited resources and might not support
some of the optimized routines this package uses. For best performance, you should
install the package on a machine that supports the `Math Kernel Library <https://en.
wikipedia.org/wiki/Math_Kernel_Library>`_.
