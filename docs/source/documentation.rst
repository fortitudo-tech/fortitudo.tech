Documentation
-------------

The left sidebar contains links to the documentation of the main modules of
this package as well as an example of how to combine views/stress-testing with
CVaR optimization.

The example code uses pandas, which is an optional package that can be installed
simultaneously with this package::

    pip install fortitudo.tech[pandas]

The data used in the example is the simulation from :cite:t:`SeqEntropyPooling`.
You can verify that the prior means and volatilities are the same by `downloading
the article using this link <https://ssrn.com/abstract_id=3936392>`_.

.. toctree::
    :hidden:

    entropy_pooling
    optimization
    examples