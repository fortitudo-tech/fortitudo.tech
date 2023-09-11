.. image:: https://github.com/fortitudo-tech/fortitudo.tech/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/fortitudo-tech/fortitudo.tech/actions/workflows/tests.yml

.. image:: https://codecov.io/gh/fortitudo-tech/fortitudo.tech/branch/main/graph/badge.svg?token=Z16XK92Gkl 
   :target: https://codecov.io/gh/fortitudo-tech/fortitudo.tech

.. image:: https://static.pepy.tech/personalized-badge/fortitudo-tech?period=total&units=international_system&left_color=grey&right_color=blue&left_text=Downloads
 :target: https://pepy.tech/project/fortitudo-tech

.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/fortitudo-tech/fortitudo.tech/main?labpath=examples

Fortitudo Technologies Open Source
==================================

This package allows you to explore open-source implementations of some of our
fundamental technologies, e.g., Entropy Pooling and CVaR optimization in Python.

The package is intended for advanced users who are comfortable specifying
portfolio constraints and Entropy Pooling views using matrices and vectors.
This gives full flexibility in relation to working with these technologies.
Hence, input checking is intentionally kept to a minimum.

If you like this package, we invite you to show your support by giving it a
GitHub star. The greater the number of stars it receives, the more dedicated
time and energy we'll invest in enhancing its functionality, providing additional
examples, and refining the code.

Fortitudo Technologies offers novel investment software as well as quantitative
and digitalization consultancy to the investment management industry. For more
information, please visit our `website <https://fortitudo.tech>`_.

Installation Instructions
-------------------------

Installation can be done via pip::

   pip install fortitudo.tech

For best performance, we recommend that you install the package in a `conda environment
<https://conda.io/projects/conda/en/latest/user-guide/concepts/environments.html>`_
and let conda handle the installation of dependencies before installing the
package using pip. You can do this by following these steps::

   conda create -n fortitudo.tech -c conda-forge python=3.11 scipy pandas matplotlib cvxopt
   conda activate fortitudo.tech
   pip install fortitudo.tech

The examples might require you to install additional packages, e.g., seaborn and
ipykernel / notebook / jupyterlab if you want to run the notebooks. Using pip to
install these packages should not cause any dependency issues.

You can also explore the examples in the cloud without any local installations using
`Binder <https://mybinder.org/v2/gh/fortitudo-tech/fortitudo.tech/main?labpath=examples>`_.
However, note that Binder servers have very limited resources and might not support
some of the optimized routines this package uses. For best performance, you should
install the package on a machine that supports the `Math Kernel Library <https://en.
wikipedia.org/wiki/Math_Kernel_Library>`_.

Disclaimer
----------

This package is completely separate from our proprietary solutions and therefore
not representative of neither the quality nor the functionality offered by the Simulation
Engine and Investment Analysis modules. If you are an institutional investor who wants
to experience how these methods can be used for sophisticated analysis in practice,
please request a demo by sending an email to demo@fortitudo.tech.
