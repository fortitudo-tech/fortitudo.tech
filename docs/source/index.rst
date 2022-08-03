Fortitudo Technologies Open Source
==================================

This package allows users to freely explore open-source implementations of some
of our fundamental technologies under the `GNU General Public License, Version 3 
<https://www.gnu.org/licenses/gpl-3.0.html>`_.

Fortitudo Technologies provides novel investment technologies as well as quantitative
and digitalization consultancy to the investment management industry. For more information,
please visit our `website <https://fortitudo.tech>`_.

This package is intended for advanced users who are comfortable specifying
portfolio constraints and Entropy Pooling views using matrices and vectors.
This gives full flexibility in relation to working with these technologies and
allows users to build their own high-level interfaces if they wish to do so.

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
   conda install -c conda-forge cvxopt=1.3 -y
   pip install fortitudo.tech

Code of Conduct
---------------

We welcome feedback and bug reports, but we have very limited resources for
support and feature requests. If you experience bugs with some of the upstream
packages, please report them directly to the maintainers of the upstream packages.

Disclaimer
----------

This package is completely separate from our proprietary solution and therefore
not representative of the functionality offered in the Investment Analysis module.
If you are an institutional investor and want to experience how these technologies
can be used in very elegant ways, please `request a demo of the Investment Analysis
module <https://fortitudo.tech/#contact>`_.

.. toctree::
   :hidden:

   documentation
   examples
   contributing
   references
