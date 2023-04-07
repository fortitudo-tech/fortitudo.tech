Installation Instructions
=========================

Installation can be done via pip::

   pip install fortitudo.tech

For best performance, we recommend that you install the package in a `conda environment
<https://conda.io/projects/conda/en/latest/user-guide/concepts/environments.html>`_
and let conda handle the installation of dependencies before installing the
package using pip. You can do this by following these steps::

   conda create -n fortitudo.tech python=3.10 scipy pandas matplotlib -y
   conda activate fortitudo.tech
   conda install -c conda-forge cvxopt=1.3 -y
   pip install fortitudo.tech

The examples might require you to install additional packages, e.g., seaborn and
ipykernel / notebook / jupyterlab if you want to run the notebooks. Using pip to
install these packages should not cause any dependency issues.

You can also explore the examples in the cloud without any local installations using
`Binder <https://mybinder.org/v2/gh/fortitudo-tech/fortitudo.tech/main?labpath=examples>`_.
However, note that Binder servers have very limited ressources and might not support
some of the optimized routines this package uses. For best performance, you should
install the package on a machine that supports the `Math Kernel Library <https://en.
wikipedia.org/wiki/Math_Kernel_Library>`_.
