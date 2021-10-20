Entropy Pooling
===============

The Entropy Pooling method solves the problem

.. math:: q=\text{argmin}\left\{ x'\left(\ln x-\ln p\right)\right\},

subject to the constraints

.. math:: Ax=b, \\Gx\leq h.

The method was first introduced by :cite:t:`EntropyPooling`, while the
code is implemented using notation from :cite:t:`SeqEntropyPooling`.

.. automodule:: fortitudo.tech.entropy_pooling
   :members:
