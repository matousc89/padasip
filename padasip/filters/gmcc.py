"""
.. versionadded:: 1.2.0

The generalized maximum correntropy criterion (GMCC)
is implemented according https://doi.org/10.1109/TSP.2016.2539127.
The GMCC adaptive filter can be created as follows

    >>> import padasip as pa
    >>> pa.filters.FilterGMCC(n)
    
where :code:`n` is the size (number of taps) of the filter.

Content of this page:

.. contents::
   :local:
   :depth: 1

.. seealso:: :ref:`filters`

Code Explanation
====================
"""
import numpy as np

from padasip.filters.base_filter import AdaptiveFilter


class FilterGMCC(AdaptiveFilter):

    kind = "GMCC"

    def __init__(self, n, mu=0.01, lambd=0.03, alpha=2, **kwargs):
        """
        This class represents an adaptive GMCC filter.

        **Kwargs:**

        * `lambd` : kernel parameter (float) commonly known as lambda.

        * `alpha` : shape parameter (float). `alpha = 2` make the filter LMS

        """
        super().__init__(mu, n, **kwargs)
        self.lambd = lambd
        self.alpha = alpha

    def learning_rule(self, e, x):
         return self.mu * self.lambd * self.alpha * \
                  np.exp(-self.lambd * (np.abs(e) ** self.alpha)) * \
                  (np.abs(e) ** (self.alpha - 1)) * np.sign(e) * x