"""
.. versionadded:: 1.2.0

The generalized maximum correntropy criterion (GMCC) adaptive filter can be created as follows

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
    """
    This class represents an adaptive GMCC filter.

    **Args:**

    * `n` : length of filter (integer) - how many input is input array
      (row of input matrix)

    **Kwargs:**

    * `mu` : learning rate (float). Also known as step size. If it is too slow,
      the filter may have bad performance. If it is too high,
      the filter will be unstable. The default value can be unstable
      for ill-conditioned input data.

    * `lambd` : kernel parameter (float) commonly known as lambda.

    * `alpha` : shape parameter (float). `alpha = 2` make the filter LMS

    * `w` : initial weights of filter. Possible values are:
        
        * array with initial weights (1 dimensional array) of filter size
    
        * "random" : create random weights
        
        * "zeros" : create zero value weights
    """
    
    def __init__(self, n, mu=0.01, lambd=0.03, alpha=2, w="random"):
        self.kind = "GMCC filter"
        if type(n) == int:
            self.n = n
        else:
            raise ValueError('The size of filter must be an integer') 
        self.mu = self.check_float_param(mu, 0, 1000, "mu")
        self.lambd = lambd
        self.alpha = alpha
        self.nu = self.mu * self.lambd * self.alpha
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
        self.w += self.nu * np.exp(-self.lambd * (np.abs(e) ** self.alpha)) * (
                    np.abs(e) ** (self.alpha - 1)) * np.sign(e) * x

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
            dw = self.nu * np.exp(-self.lambd * (np.abs(e[k]) ** self.alpha)) * (
                    np.abs(e[k]) ** (self.alpha - 1)) * np.sign(e[k]) * x[k]
            self.w += dw
        return y, e, self.w_history

