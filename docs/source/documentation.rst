Documentation
=============

Data
----

The P&L simulation from :cite:t:`SeqEntropyPooling` follows with this package
and assumes that returns follow a log-normal distribution with parameters
given by `the Danish common return expectations for the 2nd half of 2021
<https://www.afkastforventninger.dk/en/common-return-expectations/>`_.

The simulation is used in all examples and allows you to immediately start 
exploring the functionality of this package. You can also use it to test your
understanding of the theory by replicating results.

.. automodule:: fortitudo.tech.data
   :members:

Entropy Pooling
---------------

The Entropy Pooling method solves the problem

.. math:: q=\text{argmin}\left\{ x'\left(\ln x-\ln p\right)\right\},

subject to the constraints

.. math:: Ax=b, \\Gx\leq h.

The method was first introduced by :cite:t:`EntropyPooling`, while the
code is implemented using notation from :cite:t:`SeqEntropyPooling`.

.. automodule:: fortitudo.tech.entropy_pooling
   :members:

Optimization
------------

The MeanCVaR and MeanVariance objects solve the problem

.. math:: \min_{w}\mathcal{R}\left(w\right),

with :math:`\mathcal{R}\left(w\right)` being the CVaR or variance risk measure,
subject to the constraints

.. math:: \mu'w&\geq\mu_{target},\\Aw&=b,\\Gw&\leq h.

A method for solving the CVaR problem was first introduced by :cite:t:`optCVaR`,
while the implemented algorithm is based on :cite:t:`compCVaR`. The notation
in relation to the P&L simulations :math:`R` follows :cite:t:`SeqEntropyPooling`.
For the variance risk measure, a standard quadratic programming solver is used.

.. automodule:: fortitudo.tech.optimization
   :members:
   :inherited-members:

**Algorithm Parameters**

For the variance risk measure, `CVXOPT's default values 
<https://cvxopt.org/userguide/coneprog.html#algorithm-parameters>`_ are used.
These can be adjusted directly following the instructions given in the link.

For the CVaR risk measure, control parameters can be set globally using the
cvar_options dictionary, e.g.,

.. code-block:: python

   import fortitudo.tech as ft
   ft.cvar_options['demean'] = False
   ft.cvar_options['R_scalar'] = 10000

or for a particular instance of the MeanCVaR class:

.. code-block:: python

   opt = ft.MeanCVaR(R, A, b, G, h, options={'demean': False, 'R_scalar': 10000})

The following parameters can be adjusted:

:const:`'demean'`
   Whether to use demeaned P&L when calculating CVaR. Default: :const:`True`.
:const:`'R_scalar'`
   Scaling factor for the P&L simulation. Default: :const:`1000`.
:const:`'maxiter'`
   Maximum number of iterations for the decomposition algorithm, i.e., maximum
   number of relaxed master problems the algorithm is allowed to solve.
   Default: :const:`500`.
:const:`'reltol'`
   Relative tolerance for the difference between the currently best upper and
   lower bounds. Default: :const:`1e-8`.
:const:`'abstol'`
   Absolute tolerance for the difference between the currently best upper and
   lower bounds if the lower bound is less than :const:`1e-10`. Default:
   :const:`1e-8`.

The algorithm stops when one of the :const:`'maxiter'`, :const:`'reltol'`,
or :const:`'abstol'` conditions are satisfied. The parameters have been tested
with "percentage return" P&L and work well. In most cases, the algorithm stops
due to relative convergence in less than 100 iterations. But if you use P&L
simulations that are scaled differently, you might need to adjust them.
