"""
.. versionadded:: 0.2
.. versionchanged:: 1.2.0

The generalized normalized gradient descent (GNGD) adaptive filter
is an extension of the NLMS adaptive filter (:ref:`filter-nlms`).

The GNGD filter can be created as follows

    >>> import padasip as pa
    >>> pa.filters.FilterGNGD(n)

where `n` is the size (number of taps) of the filter.

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
    f = pa.filters.FilterGNGD(n=4, mu=0.1, w="random")
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
======================================
"""
import numpy as np

from padasip.filters.base_filter import AdaptiveFilter

class FilterGNGD(AdaptiveFilter):
    """
    Adaptive GNGD filter.
    """
    kind = "GNGD"

    def __init__(self, n, mu=1., eps=1., ro=0.1, **kwargs):
        """
        **Kwargs:**

        * `eps` : compensation term (float) at the beginning. It is adaptive
          parameter.

        * `ro` : step size adaptation parameter (float) at the beginning.
          It is adaptive parameter.

        """
        super().__init__(n, mu, **kwargs)
        self.eps = eps
        self.ro = ro
        self.last_e = 0
        self.last_x = np.zeros(n)

    def learning_rule(self, e, x):
        """
        Override the parent class.
        """
        self.eps = self.eps - self.ro * self.mu * e * self.last_e * \
                   np.dot(x, self.last_x) / \
                   (np.dot(self.last_x, self.last_x) + self.eps) ** 2
        nu = self.mu / (self.eps + np.dot(x, x))
        self.last_e, self.last_x = e, x
        return nu * e * x
