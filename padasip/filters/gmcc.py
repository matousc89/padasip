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

Minimal Working Examples
======================================

If you have measured data you may filter it as follows

.. code-block:: python

    import numpy as np
    import matplotlib.pylab as plt
    import padasip as pa

    # creation of data
    N = 500
    x = np.random.normal(0, 1, (N, 4)) # input matrix
    v = np.random.normal(0, 0.1, N) # noise
    d = 2*x[:,0] + 0.1*x[:,1] - 4*x[:,2] + 0.5*x[:,3] + v # target

    # identification
    f = pa.filters.FilterGMCC(n=4, mu=0.1, w="random")
    y, e, w = f.run(d, x)

    # show results
    plt.figure(figsize=(15,9))
    plt.subplot(211);plt.title("Adaptation");plt.xlabel("samples - k")
    plt.plot(d,"b", label="d - target")
    plt.plot(y,"g", label="y - output");plt.legend()
    plt.subplot(212);plt.title("Filter error");plt.xlabel("samples - k")
    plt.plot(10*np.log10(e**2),"r", label="e - error [dB]");plt.legend()
    plt.tight_layout()
    plt.show()

Code Explanation
====================
"""
import numpy as np

from padasip.filters.base_filter import AdaptiveFilter


class FilterGMCC(AdaptiveFilter):
    """
    This class represents an adaptive GMCC filter.
    """
    kind = "GMCC"

    def __init__(self, n, mu=0.01, lambd=0.03, alpha=2, **kwargs):
        """
        **Kwargs:**

        * `lambd` : kernel parameter (float) commonly known as lambda.

        * `alpha` : shape parameter (float). `alpha = 2` make the filter LMS
        """
        super().__init__(n, mu, **kwargs)
        self.lambd = lambd
        self.alpha = alpha

    def learning_rule(self, e, x):
        """
        Override the parent class.
        """
        return self.mu * self.lambd * self.alpha * \
            np.exp(-self.lambd * (np.abs(e) ** self.alpha)) * \
            (np.abs(e) ** (self.alpha - 1)) * np.sign(e) * x
