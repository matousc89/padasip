"""
.. versionadded:: 1.1.0
.. versionchanged:: 1.2.0

The normalized sign-sign least-mean-squares (NSSLMS) adaptive filter
is an extension of the popular SSLMS adaptive filter (:ref:`filter-sslms`).

The NSSLMS filter can be created as follows

    >>> import padasip as pa
    >>> pa.filters.FilterNSSLMS(n)

where `n` is the size (number of taps) of the filter.

Content of this page:

.. contents::
   :local:
   :depth: 1

.. seealso:: :ref:`filters`

Algorithm Explanation
======================================

The NSSLMS is extension of LMS filter. See :ref:`filter-lms`
for explanation of the algorithm behind.

The extension is based on normalization of learning rate.
The learning rage :math:`\mu` is replaced by learning rate :math:`\eta(k)`
normalized with every new sample according to input power as follows

:math:`\eta (k) = \\frac{\mu}{\epsilon + || \\textbf{x}(k) ||^2}`,

where :math:`|| \\textbf{x}(k) ||^2` is norm of input vector and
:math:`\epsilon` is a small positive constant (regularization term).
This constant is introduced to preserve the stability in cases where
the input is close to zero.

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
    d = 2*x[:,0] + 0.1*x[:,1] - 0.3*x[:,2] + 0.5*x[:,3] + v # target

    # identification
    f = pa.filters.FilterNSSLMS(n=4, mu=0.1, w="random")
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

class FilterNSSLMS(AdaptiveFilter):
    """
    Adaptive NSSLMS filter.
    """
    kind = "NSSLMS"

    def __init__(self, n, mu=0.1, eps=0.001, **kwargs):
        """
        **Kwargs:**

        * `eps` : regularization term (float). It is introduced to preserve
          stability for close-to-zero input vectors
        """
        super().__init__(n, mu, **kwargs)
        self.eps = eps

    def learning_rule(self, e, x):
        """
        Override the parent class.
        """
        return self.mu / (self.eps + np.dot(x, x)) * np.sign(x) * np.sign(e)
