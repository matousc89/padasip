"""
.. versionadded:: 0.1
.. versionchanged:: 1.2.0

The Recursive Least Squares filter can be created as follows

    >>> import padasip as pa
    >>> pa.filters.FilterRLS(n)

where the n is amount of filter inputs (size of input vector).

Content of this page:

.. contents::
   :local:
   :depth: 1

.. seealso:: :ref:`filters`

Algorithm Explanation
======================================

The RLS adaptive filter may be described as

:math:`y(k) = w_1 \cdot x_{1}(k) + ... + w_n \cdot x_{n}(k)`,

or in a vector form

:math:`y(k) = \\textbf{x}^T(k) \\textbf{w}(k)`,

where :math:`k` is discrete time index, :math:`(.)^T` denotes the transposition,
:math:`y(k)` is filtered signal,
:math:`\\textbf{w}` is vector of filter adaptive parameters and
:math:`\\textbf{x}` is input vector (for a filter of size :math:`n`) as follows

:math:`\\textbf{x}(k) = [x_1(k), ...,  x_n(k)]`.

The update is done as folows

:math:`\\textbf{w}(k+1) = \\textbf{w}(k) + \Delta \\textbf{w}(k)`

where :math:`\Delta \\textbf{w}(k)` is obtained as follows

:math:`\Delta \\textbf{w}(k) = \\textbf{R}(k) \\textbf{x}(k) e(k)`,

where :math:`e(k)` is error and it is estimated according to filter output
and desired value :math:`d(k)` as follows

:math:`e(k) = d(k) - y(k)`

The :math:`\\textbf{R}(k)` is inverse of autocorrelation matrix
and it is calculated as follows

:math:`\\textbf{R}(k) = \\frac{1}{\\mu}(
\\textbf{R}(k-1) -
\\frac{\\textbf{R}(k-1)\\textbf{x}(k) \\textbf{x}(k)^{T} \\textbf{R}(k-1)}
{\\mu + \\textbf{x}(k)^{T}\\textbf{R}(k-1)\\textbf{x}(k)}
)`.

The initial value of autocorrelation matrix should be set to

:math:`\\textbf{R}(0) = \\frac{1}{\\delta} \\textbf{I}`,

where :math:`\\textbf{I}` is identity matrix and :math:`\delta`
is small positive constant.

Stability and Optimal Performance
======================================

Make the RLS working correctly with a real data can be tricky.
The forgetting factor :math:`\\mu` should be in range from 0 to 1.
But in a lot of cases it works only with values close to 1
(for example something like 0.99).

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
    f = pa.filters.FilterRLS(n=4, mu=0.1, w="random")
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
    filt = pa.filters.FilterRLS(3, mu=0.5)
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

class FilterRLS(AdaptiveFilter):
    """
    Adaptive RLS filter.
    """
    kind = "RLS"

    def __init__(self, n, mu=0.1, eps=0.001, **kwargs):
        """
        **Kwargs:**

        * `eps` : initialisation value (float). It is usually chosen
          between 0.1 and 1.
        """
        super().__init__(n, mu, **kwargs)
        self.eps = eps
        self.R = 1 / self.eps * np.identity(n)

    def learning_rule(self, e, x):
        """
        Override the parent class.
        """
        R1 = self.R @ (x[:, None] * x[None, :]) @ self.R
        R2 = self.mu + np.dot(np.dot(x, self.R), x.T)
        self.R = 1 / self.mu * (self.R - R1/R2)
        return np.dot(self.R, x.T) * e
