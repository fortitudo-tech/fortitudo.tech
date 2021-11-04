Fortitudo Technologies Open Source
==================================

This package allows users to freely explore open-source implementations of some
of our fundamental technologies under the `GNU General Public License, Version 3 
<https://www.gnu.org/licenses/gpl-3.0.html>`_.

Fortitudo Technologies is a fintech company offering novel software solutions
as well as quantitative and digitalization consultancy to the investment 
management industry. For more information, please visit our `website 
<https://fortitudo.tech>`_.

This package is intended for advanced users who are comfortable specifying
portfolio constraints and Entropy Pooling views using matrices and vectors.
This gives users full flexibility in relation to working with these technologies
and allows them to build their own high-level interfaces if they wish to do so.

Installation Instructions
-------------------------

Installation can be done via pip::

   pip install fortitudo.tech

For best performance, we recommend that you install the package in a `conda environment
<https://conda.io/projects/conda/en/latest/user-guide/concepts/environments.html>`_
and let conda handle the installation of dependencies before installing the
package using pip. You can do this by following these steps::

   conda create -n fortitudo.tech python scipy pandas -y
   conda activate fortitudo.tech
   conda install -c conda-forge cvxopt=1.2.6 -y
   pip install fortitudo.tech

Code of Conduct
---------------

We welcome feedback and bug reports, but we have very limited resources for
support and feature requests.

If you experience bugs with some of the upstream packages, please report these
directly to the maintainers of the upstream packages.

Disclaimer
----------

This package is completely separate from our proprietary solution and therefore
not representative of the functionality offered therein.

.. toctree::
   :hidden:

   documentation
   examples
   contributing
   references
