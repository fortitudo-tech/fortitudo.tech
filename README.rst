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
For a high-level introduction, see this `YouTube video <https://youtu.be/FOiY8aRVC90>`_
and `Medium article <https://medium.com/@ft_anvo/entropy-pooling-and-cvar-portfolio-optimization-in-python-ffed736a8347>`_.

The package is intended for advanced users who are comfortable specifying
portfolio constraints and Entropy Pooling views using matrices and vectors.
This gives full flexibility in relation to working with these technologies.
Hence, input checking is intentionally kept to a minimum.

If you like this package, we invite you to show your support by giving it a
GitHub star. The greater the number of stars it receives, the more time and
energy we will invest in enhancing its functionality and providing additional
examples.

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

   conda create -n fortitudo.tech -c conda-forge python scipy pandas matplotlib cvxopt
   conda activate fortitudo.tech
   pip install fortitudo.tech

The examples might require you to install additional packages, e.g., seaborn and
ipykernel / notebook / jupyterlab if you want to run the notebooks. Using pip to
install these packages should not cause any dependency issues.

You can also explore the examples in the cloud without any local installations using
`Binder <https://mybinder.org/v2/gh/fortitudo-tech/fortitudo.tech/main?labpath=examples>`_.
However, note that Binder servers have very limited resources and might not support
some of the optimized routines this package uses. If you want access to a stable
and optimized environment with persistent storage, please subscribe to our Data
Science Server.

Disclaimer
----------

This package is completely separate from our proprietary solutions and therefore
not representative of the quality and functionality offered by the Investment Simulation
and Investment Analysis modules. If you are an institutional investor who wants to
experience how these methods can be used for sophisticated analysis in practice,
please request a demo by sending an email to demo@fortitudo.tech.
