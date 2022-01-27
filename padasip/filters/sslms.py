"""
.. versionadded:: 1.1.0
.. versionchanged:: 1.2.0

The sign-sign least-mean-squares (SSLMS) adaptive filter can be created as follows

    >>> import padasip as pa
    >>> pa.filters.FilterSSLMS(n)

where :code:`n` is the size (number of taps) of the filter.

Content of this page:

.. contents::
   :local:
   :depth: 1

.. seealso:: :ref:`filters`

Algorithm Explanation
==========================

The SSLMS adaptive filter could be described as

:math:`y(k) = w_1 \cdot x_{1}(k) + ... + w_n \cdot x_{n}(k)`,

or in a vector form

:math:`y(k) = \\textbf{x}^T(k) \\textbf{w}(k)`,

where :math:`k` is discrete time index, :math:`(.)^T` denotes the transposition,
:math:`y(k)` is filtered signal,
:math:`\\textbf{w}` is vector of filter adaptive parameters and
:math:`\\textbf{x}` is input vector (for a filter of size :math:`n`) as follows

:math:`\\textbf{x}(k) = [x_1(k), ...,  x_n(k)]`.

The SSLMS weights adaptation could be described as follows

:math:`\\textbf{w}(k+1) = \\textbf{w}(k) + \Delta \\textbf{w}(k)`,

where :math:`\Delta \\textbf{w}(k)` is

:math:`\Delta \\textbf{w}(k) =  \mu \cdot \\text{sgn}(e(k)) \cdot
\\text{sgn}(\\textbf{x}(k))`,

where :math:`\mu` is the learning rate (step size) and :math:`e(k)`
is error defined as

:math:`e(k) = d(k) - y(k)`.


Minimal Working Examples
==============================

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
    f = pa.filters.FilterSSLMS(n=4, mu=0.01, w="random")
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


class FilterSSLMS(AdaptiveFilter):
    """
    This class represents an adaptive SSLMS filter.
    """
    kind = "SSLMS"

    def learning_rule(self, e, x):
        """
        Override the parent class.
        """
        return self.mu * np.sign(x) * np.sign(e)
