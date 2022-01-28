"""
.. versionadded:: 0.4
.. versionchanged:: 1.2.0

The Affine Projection (AP) algorithm is implemented according to paper.
Usage of this filter should be benefical especially
when input data is highly correlated.
This filter is based on LMS. The difference is,
that AP uses multiple input vectors in every sample.
The number of vectors is called projection order.
In this implementation the historic input vectors from input matrix are used
as the additional input vectors in every sample.

The AP filter can be created as follows

    >>> import padasip as pa
    >>> pa.filters.FilterAP(n)

where `n` is the size of the filter.

Content of this page:

.. contents::
   :local:
   :depth: 1

.. seealso:: :ref:`filters`

Algorithm Explanation
======================================

The input for AP filter is created as follows

:math:`\\textbf{X}_{AP}(k) = (\\textbf{x}(k), ..., \\textbf{x}(k-L))`,

where :math:`\\textbf{X}_{AP}` is filter input, :math:`L` is projection order,
:math:`k` is discrete time index and :math:`\\textbf{x}_{k}` is input vector.
The output of filter si calculated as follows:

:math:`\\textbf{y}_{AP}(k) = \\textbf{X}^{T}_{AP}(k) \\textbf{w}(k)`,

where :math:`\\textbf{x}(k)` is the vector of filter adaptive parameters.
The vector of targets is constructed as follows

:math:`\\textbf{d}_{AP}(k) = (d(k), ..., d(k-L))^T`,

where :math:`d(k)` is target in time :math:`k`.

The error of the filter is estimated as

:math:`\\textbf{e}_{AP}(k) = \\textbf{d}_{AP}(k) - \\textbf{y}_{AP}(k)`.

And the adaptation of adaptive parameters is calculated according to equation

:math:`\\textbf{w}_{AP}(k+1) =
\\textbf{w}_{AP}(k+1) + \mu \\textbf{X}_{AP}(k) (\\textbf{X}_{AP}^{T}(k)
\\textbf{X}_{AP}(k) + \epsilon \\textbf{I})^{-1} \\textbf{e}_{AP}(k)`.

During the filtering we are interested just in output of filter :math:`y(k)`
and the error :math:`e(k)`. These two values are the first elements in
vectors: :math:`\\textbf{y}_{AP}(k)` for output and
:math:`\\textbf{e}_{AP}(k)` for error.



Minimal Working Example
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
    f = pa.filters.FilterAP(n=4, order=5, mu=0.5, ifc=0.001, w="random")
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
    filt = pa.filters.FilterAP(3, mu=1.)
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

from padasip.filters.base_filter import AdaptiveFilterAP

class FilterAP(AdaptiveFilterAP):
    """
    This class represents an adaptive AP filter.
    """
    kind = "AP"

    def learning_rule(self, e_mem, x_mem):
        """
        Override the parent class.
        """
        dw_part1 = np.dot(x_mem.T, x_mem) + self.ide_ifc
        dw_part2 = np.linalg.solve(dw_part1, self.ide)
        return self.mu * np.dot(x_mem, np.dot(dw_part2, e_mem))
