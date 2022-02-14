Contributing
============

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
imposing object-oriented design patterns. We are unlikely to accept contributions
that use inheritance without very good reasons and encourage you to use composition
instead.