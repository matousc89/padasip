"""
.. versionadded:: 1.1.0

The least-mean-fourth (LMF) adaptive filter implemented according to the
paper :cite:`zerguine2000convergence`.

The LMF filter can be created as follows

    >>> import padasip as pa
    >>> pa.filters.FilterLMF(n)
    
where :code:`n` is the size (number of taps) of the filter.

Content of this page:

.. contents::
   :local:
   :depth: 1

.. seealso:: :ref:`filters`

Algorithm Explanation
==========================

The LMF adaptive filter could be described as

:math:`y(k) = w_1 \cdot x_{1}(k) + ... + w_n \cdot x_{n}(k)`,

or in a vector form

:math:`y(k) = \\textbf{x}^T(k) \\textbf{w}(k)`,

where :math:`k` is discrete time index, :math:`(.)^T` denotes the transposition,
:math:`y(k)` is filtered signal,
:math:`\\textbf{w}` is vector of filter adaptive parameters and
:math:`\\textbf{x}` is input vector (for a filter of size :math:`n`) as follows

:math:`\\textbf{x}(k) = [x_1(k), ...,  x_n(k)]`.

The LMF weights adaptation could be described as follows

:math:`\\textbf{w}(k+1) = \\textbf{w}(k) + \Delta \\textbf{w}(k)`,

where :math:`\Delta \\textbf{w}(k)` is

:math:`\Delta \\textbf{w}(k) = \\frac{1}{2} \mu \\frac{\partial e^4(k)}
{ \partial \\textbf{w}(k)}\ = \mu \cdot e(k)^{3} \cdot \\textbf{x}(k)`,

where :math:`\mu` is the learning rate (step size) and :math:`e(k)`
is error defined as

:math:`e(k) = d(k) - y(k)`.


Minimal Working Examples
==============================

If you have measured data you may filter it as follows

.. code-block:: python

    # creation of data
    N = 500
    x = np.random.normal(0, 1, (N, 4)) # input matrix
    v = np.random.normal(0, 0.1, N) # noise
    d = 2*x[:,0] + 0.1*x[:,1] - 4*x[:,2] + 0.5*x[:,3] + v # target

    # identification
    f = pa.filters.FilterLMF(n=4, mu=0.01, w="random")
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

References
============

.. bibliography:: lmf.bib
    :style: plain

Code Explanation
====================
"""
import numpy as np

from padasip.filters.base_filter import AdaptiveFilter

class FilterLMF(AdaptiveFilter):
    """
    This class represents an adaptive LMF filter.

    **Args:**

    * `n` : length of filter (integer) - how many input is input array
      (row of input matrix)

    **Kwargs:**

    * `mu` : learning rate (float). Also known as step size. If it is too slow,
      the filter may have bad performance. If it is too high,
      the filter will be unstable. The default value can be unstable
      for ill-conditioned input data.

    * `w` : initial weights of filter. Possible values are:
        
        * array with initial weights (1 dimensional array) of filter size
    
        * "random" : create random weights
        
        * "zeros" : create zero value weights
    """
    
    def __init__(self, n, mu=0.01, w="random"):
        self.kind = "LMF filter"
        if type(n) == int:
            self.n = n
        else:
            raise ValueError('The size of filter must be an integer') 
        self.mu = self.check_float_param(mu, 0, 1000, "mu")
        self.init_weights(w, self.n)
        self.w_history = False

    def adapt(self, d, x):
        """
        Adapt weights according one desired value and its input.

        **Args:**

        * `d` : desired value (float)

        * `x` : input array (1-dimensional array)
        """
        y = np.dot(self.w, x)
        e = d - y
        self.w += self.mu * x * e**3       

    def run(self, d, x):
        """
        This function filters multiple samples in a row.

        **Args:**

        * `d` : desired value (1 dimensional array)

        * `x` : input matrix (2-dimensional array). Rows are samples,
          columns are input arrays.

        **Returns:**

        * `y` : output value (1 dimensional array).
          The size corresponds with the desired value.

        * `e` : filter error for every sample (1 dimensional array).
          The size corresponds with the desired value.

        * `w` : history of all weights (2 dimensional array).
          Every row is set of the weights for given sample.
        """
        # measure the data and check if the dimmension agree
        N = len(x)
        if not len(d) == N:
            raise ValueError('The length of vector d and matrix x must agree.')  
        self.n = len(x[0])
        # prepare data
        try:    
            x = np.array(x)
            d = np.array(d)
        except:
            raise ValueError('Impossible to convert x or d to a numpy array')
        # create empty arrays
        y = np.zeros(N)
        e = np.zeros(N)
        self.w_history = np.zeros((N,self.n))
        # adaptation loop
        for k in range(N):
            self.w_history[k,:] = self.w
            y[k] = np.dot(self.w, x[k])
            e[k] = d[k] - y[k]
            dw = self.mu * x[k] * e[k]**3
            self.w += dw
        return y, e, self.w_history

