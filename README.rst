.. image:: https://github.com/fortitudo-tech/fortitudo.tech/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/fortitudo-tech/fortitudo.tech/actions/workflows/tests.yml

.. image:: https://codecov.io/gh/fortitudo-tech/fortitudo.tech/branch/main/graph/badge.svg?token=Z16XK92Gkl 
   :target: https://codecov.io/gh/fortitudo-tech/fortitudo.tech

.. image:: https://static.pepy.tech/personalized-badge/fortitudo-tech?period=total&units=international_system&left_color=grey&right_color=blue&left_text=Downloads
   :target: https://pepy.tech/project/fortitudo-tech

Fortitudo Technologies Open Source
==================================

This package allows you to freely explore open-source implementations of some
of our fundamental technologies under the `GNU General Public License, Version 3
<https://www.gnu.org/licenses/gpl-3.0.html>`_.

The package is intended for advanced users who are comfortable specifying
portfolio constraints and Entropy Pooling views using matrices and vectors.
This gives full flexibility in relation to working with these technologies
and allows you to build your own high-level interfaces if you want. Hence,
input checking is intentionally kept to a minimum.

Fortitudo Technologies is a fintech company offering novel investment technologies
as well as quantitative and digitalization consultancy to the investment management
industry. For more information, please visit our `website <https://fortitudo.tech>`_.

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

Contributing
------------

You are welcome to contribute to this package by forking the `fortitudo.tech 
GitHub repository <https://github.com/fortitudo-tech/fortitudo.tech>`_ and
creating pull requests. Pull requests should always be sent to the dev branch.
We especially appreciate contributions in relation to packaging, e.g., making
the package available on conda-forge.

Using the conda environment specified in the requirements.yml file and located
in the root directory of the repository is the easiest way to start contributing
to the code::

    conda env create --file requirements.yml

The style guide mostly follows `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_,
but it uses some important modifications that can be found in .vscode/settings.json.
If you use Visual Studio Code, you can use these settings to make sure that
your code follows the basic rules of the style guide. The most important
modifications/additions are:

1) We allow line length to be 99 characters for both code and docstrings,
2) We allow the use of capital I as a variable,
3) We use type hints introduced in `PEP 484 <https://www.python.org/dev/peps/pep-0484/>`_,
4) We do not group operators according to priority.

We generally follow naming conventions with descriptive variable and function
names, but we often use short variable names for the very mathematical parts of
the code to replicate the variables used in the references. We believe this makes
it easier to link the code to the theory.

We encourage you to keep individual contributions small in addition to avoid
imposing object-oriented design patterns.

Code of Conduct
---------------

We welcome feedback and bug reports, but we have very limited resources for
support and feature requests. If you experience bugs with some of the upstream
packages, please report them directly to the maintainers of the upstream packages.

Disclaimer
----------

This package is completely separate from our proprietary solutions and therefore
not representative of the functionality offered therein. The examples for this
package illustrate only elementary use cases. If you are an institutional investor
and want to experience how these technologies can be used in more sophisticated
ways, please request a demo by sending an email to demo@fortitudo.tech.
