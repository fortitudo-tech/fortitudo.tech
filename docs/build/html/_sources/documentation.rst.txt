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

The MeanCVaR object can solve the problem

.. math:: \min_{e}CVaR\left(R,p,\alpha,e\right),

subject to the constraints

.. math:: \mu'e&\geq\mu_{target},\\Ae&=b,\\Ge&\leq h.

A method for solving this problem was first introduced by :cite:t:`optCVaR`,
while the implemented algorithm is based on :cite:t:`compCVaR`. The notation
in relation to the P&L simulations :math:`R` follows :cite:t:`SeqEntropyPooling`.

.. automodule:: fortitudo.tech.optimization
   :members:

**Algorithm Parameters**

Control parameters can be set globally using the cvar_options dictionary, e.g.,

.. code-block:: python

   import fortitudo.tech as ft
   ft.cvar_options['demean'] = False
   ft.cvar_options['R_scalar'] = 10000

or for a particular instance of the MeanCVaR class:

.. code-block:: python

   opt = ft.MeanCVaR(R, A, b, G, h, options={'demean': False, 'R_scalar': 10000})

The following parameters can be accessed:

:const:`'demean'`
   Whether to use demeaned P&L when calculating CVaR. Default: :const:`True`.
:const:`'R_scalar'`
   Scaling factor for the P&L simulation. Default: :const:`1000`.
:const:`'mean_scalar'`
   Scaling factor for the expected returns used by the return target constraint.
   Default: :const:`100`.
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
with P&L that is "percentage return" and work well. In most cases, the algorithm
stops due to relative convergence in less than 100 iterations. But if you use
P&L simulations that are scaled differently, you might need to adjust them.
