"""
.. versionadded:: 1.2.0

The least lncosh (Llncosh) algorithm (proposed in https://doi.org/10.1016/j.sigpro.2019.107348)
is similar to LMS adaptive filter.

The Llncosh filter can be created as follows

    >>> import padasip as pa
    >>> pa.filters.FilterLlncosh(n)
    
where :code:`n` is the size (number of taps) of the filter.

Content of this page:

.. contents::
   :local:
   :depth: 1

.. seealso:: :ref:`filters`

Algorithm Explanation
==========================

The lncosh cost function is the natural logarithm of hyperbolic cosine function,
which behaves like a hybrid of the mean square error and mean absolute error
criteria according to its positive parameter `l`.

Minimal Working Examples
==============================

If you have measured data you may filter it as follows

.. code-block:: python

    import numpy as np
    import matplotlib.pylab as plt
    import padasip as pa

    # creation of data
    N = 500
    x = np.random.normal(0, 1, (N, 4))  # input matrix
    v = np.random.normal(0, 0.1, N)  # noise
    d = 2 * x[:, 0] + 0.1 * x[:, 1] - 4 * x[:, 2] + 0.5 * x[:, 3] + v  # target

    # identification
    f = pa.filters.FilterLlncosh(n=4, mu=0.1, l=0.1, w="random")
    y, e, w = f.run(d, x)

    # show results
    plt.figure(figsize=(15, 9))
    plt.subplot(211);
    plt.title("Adaptation");
    plt.xlabel("samples - k")
    plt.plot(d, "b", label="d - target")
    plt.plot(y, "g", label="y - output");
    plt.legend()
    plt.subplot(212);
    plt.title("Filter error");
    plt.xlabel("samples - k")
    plt.plot(10 * np.log10(e ** 2), "r", label="e - error [dB]");
    plt.legend()
    plt.tight_layout()
    plt.show()

Code Explanation
====================
"""
import numpy as np

from padasip.filters.base_filter import AdaptiveFilter


class FilterLlncosh(AdaptiveFilter):
    """
    This class represents an adaptive Llncosh filter.

    **Args:**

    * `n` : length of filter (integer) - how many input is input array
      (row of input matrix)

    **Kwargs:**

    * `mu` : learning rate (float). Also known as step size. If it is too slow,
      the filter may have bad performance. If it is too high,
      the filter will be unstable. The default value can be unstable
      for ill-conditioned input data.

    * `mu` : lambda (float). Cost function shape parameter.

    * `w` : initial weights of filter. Possible values are:
        
        * array with initial weights (1 dimensional array) of filter size
    
        * "random" : create random weights
        
        * "zeros" : create zero value weights
    """
    kind = "Llncosh"

    def __init__(self, n, mu=0.01, l=3, w="random"):
        if type(n) == int:
            self.n = n
        else:
            raise ValueError('The size of filter must be an integer')
        self.mu = self.check_float_param(mu, 0, 1000, "mu")
        self.l = l
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
        self.w += self.mu * np.tanh(self.l * e) * x

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
            dw = self.mu * np.tanh(self.l * e[k]) * x[k]
            self.w += dw
        return y, e, self.w_history

