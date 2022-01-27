"""
.. versionadded:: 0.1
.. versionchanged:: 1.2.0

The normalized least-mean-squares (NLMS) adaptive filter
is an extension of the popular LMS adaptive filter (:ref:`filter-lms`).

The NLMS filter can be created as follows

    >>> import padasip as pa
    >>> pa.filters.FilterNLMS(n)

where `n` is the size (number of taps) of the filter.

Content of this page:

.. contents::
   :local:
   :depth: 1

.. seealso:: :ref:`filters`

Algorithm Explanation
======================================

The NLMS is extension of LMS filter. See :ref:`filter-lms`
for explanation of the algorithm behind.

The extension is based on normalization of learning rate.
The learning rage :math:`\mu` is replaced by learning rate :math:`\eta(k)`
normalized with every new sample according to input power as follows

:math:`\eta (k) = \\frac{\mu}{\epsilon + || \\textbf{x}(k) ||^2}`,

where :math:`|| \\textbf{x}(k) ||^2` is norm of input vector and
:math:`\epsilon` is a small positive constant (regularization term).
This constant is introduced to preserve the stability in cases where
the input is close to zero.

Stability and Optimal Performance
======================================

The stability of the NLMS filter si given as follows

:math:`0 \le \mu \le 2 + \\frac{2\epsilon}{||\\textbf{x}(k)||^2}`,

or in case without regularization term :math:`\epsilon`

:math:`\mu \in <0, 2>`.

In other words, if you use the zero or only small key argument `\eps`,
the key argument `\mu` should be between 0 and 2. Best convergence
should be produced by `mu=1.` according to theory. However in practice
the optimal value can be strongly case specific.


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
    f = pa.filters.FilterNLMS(n=4, mu=0.1, w="random")
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

An example how to filter data measured in real-time

.. code-block:: python

    import numpy as np
    import matplotlib.pylab as plt
    import padasip as pa

    # these two function supplement your online measurment
    def measure_x():
        # it produces input vector of size 3
        x = np.random.random(3)
        return x

    def measure_d(x):
        # meausure system output
        d = 2*x[0] + 1*x[1] - 1.5*x[2]
        return d

    N = 100
    log_d = np.zeros(N)
    log_y = np.zeros(N)
    filt = pa.filters.FilterNLMS(3, mu=1.)
    for k in range(N):
        # measure input
        x = measure_x()
        # predict new value
        y = filt.predict(x)
        # do the important stuff with prediction output
        pass
        # measure output
        d = measure_d(x)
        # update filter
        filt.adapt(d, x)
        # log values
        log_d[k] = d
        log_y[k] = y

    ### show results
    plt.figure(figsize=(15,9))
    plt.subplot(211);plt.title("Adaptation");plt.xlabel("samples - k")
    plt.plot(log_d,"b", label="d - target")
    plt.plot(log_y,"g", label="y - output");plt.legend()
    plt.subplot(212);plt.title("Filter error");plt.xlabel("samples - k")
    plt.plot(10*np.log10((log_d-log_y)**2),"r", label="e - error [dB]")
    plt.legend(); plt.tight_layout(); plt.show()



Code Explanation
======================================
"""
import numpy as np

from padasip.filters.base_filter import AdaptiveFilter

class FilterNLMS(AdaptiveFilter):
    """
    Adaptive NLMS filter.
    """
    kind = "NLMS"

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
        return self.mu / (self.eps + np.dot(x, x)) * x * e
