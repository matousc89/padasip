"""
.. versionadded:: 0.2

The generalized normalized gradient descent (GNGD) adaptive filter
:cite:`mandic2004generalized`
is an extension of the NLMS adaptive filter (:ref:`filter-nlms-label`).

The GNGD filter can be created as follows

    >>> import padasip as pa
    >>> pa.filters.FilterGNGD(n)
    
where `n` is the size (number of taps) of the filter.

Content of this page:

.. contents::
   :local:
   :depth: 1
   
References
***************

.. bibliography:: gngd.bib
    :style: plain

Code Explanation
***************** 
"""
import numpy as np
import padasip.consts as co

from padasip.filters.base_filter import AdaptiveFilter


class FilterGNGD(AdaptiveFilter):
    """
    Adaptive GNGD filter.

    Based on:

    Mandic, D. P. (2004). A generalized normalized gradient descent algorithm.
    Signal Processing Letters, IEEE, 11(2), 115-118.

    Args:

    * `n` : length of filter (integer) - how many input is input array
        (row of input matrix)

    Kwargs:

    * `mu` : learning rate (float). Also known as step size.
        If it is too slow,
        the filter may have bad performance. If it is too high,
        the filter will be unstable. The default value can be unstable
        for ill-conditioned input data.

    * `eps` : compensation term (float) at the beginning. It is adaptive
        parameter.

    * `ro` : step size adaptation parameter (float) at the beginning.
        It is adaptive parameter.

    * `w` : initial weights of filter. Possible values are:
        
        * array with initial weights (1 dimensional array) of filter size
    
        * "random" : create random weights
        
        * "zeros" : create zero value weights
    """ 
    def __init__(self, n, mu=co.MU_GNGD, eps=co.EPS_GNGD,
            ro=co.RO_GNGD, w="random",):
        self.kind = "NLMS filter"
        if type(n) == int:
            self.n = n
        else:
            raise ValueError('The size of filter must be an integer') 
        self.mu = self.check_float_param(mu, co.MU_GNGD_MIN, co.MU_GNGD_MAX, "mu")
        self.eps = self.check_float_param(eps, co.EPS_GNGD_MIN,
            co.EPS_GNGD_MAX, "eps")
        self.ro = self.check_float_param(ro, co.RO_GNGD_MIN,
            co.RO_GNGD_MAX, "ro")
        self.last_e = 0
        self.last_x = np.zeros(n)
        self.w = self.init_weights(w, self.n)
        self.w_history = False

    def adapt(self, d, x):
        """
        Adapt weights according one desired value and its input.

        Args:

        * `d` : desired value (float)

        * `x` : input array (1-dimensional array)
        """
        y = np.dot(self.w, x)
        e = d - y
        self.eps = self.eps - self.ro * self.mu * e * self.last_e * \
            np.dot(x, self.last_x) / \
            (np.dot(self.last_x, self.last_x) + self.eps)**2
        nu = self.mu / (self.eps + np.dot(x, x))
        self.w += nu * e * x        
        self.last_e = e

    def run(self, d, x):
        """
        This function filters multiple samples in a row.

        Args:

        * `d` : desired value (1 dimensional array)

        * `x` : input matrix (2-dimensional array). Rows are samples, columns are
            input arrays.

        Returns:

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
        self.w_history = np.zeros((N, self.n))
        # adaptation loop
        for k in range(N):
            y[k] = np.dot(self.w, x[k])
            e[k] = d[k] - y[k]
            self.eps = self.eps - self.ro * self.mu * e[k] * e[k-1] * \
                np.dot(x[k], x[k-1]) / \
                (np.dot(x[k-1], x[k-1]) + self.eps)**2
            nu = self.mu / (self.eps + np.dot(x[k], x[k]))
            dw = nu * e[k] * x[k]
            self.w += dw
            self.w_history[k:] = self.w
        return y, e, self.w
        
    def novelty(self, d, x):
        """
        This function estimates novelty in data
        according to the learning effort.

        Args:

        * `d` : desired value (1 dimensional array)

        * `x` : input matrix (2-dimensional array). Rows are samples,
            columns are input arrays.

        Returns:

        * `y` : output value (1 dimensional array).
            The size corresponds with the desired value.

        * `e` : filter error for every sample (1 dimensional array). 
            The size corresponds with the desired value.

        * `w` : history of all weights (2 dimensional array).
            Every row is set of the weights for given sample.

        * `nd` : novelty detection coefficients (2 dimensional array).
            Every row is set of coefficients for given sample.
            One coefficient represents one filter weight.
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
        nd = np.zeros((N,self.n))
        w_all = np.zeros((N,self.n))
        self.w_history = np.zeros((N, self.n))
        # adaptation loop
        for k in range(N):
            y[k] = np.dot(self.w, x[k])
            e[k] = d[k] - y[k]
            self.eps = self.eps - self.ro * self.mu * e[k] * e[k-1] * \
                np.dot(x[k], x[k-1]) / \
                (np.dot(x[k-1], x[k-1]) + self.eps)**2
            nu = self.mu / (self.eps + np.dot(x[k], x[k]))
            dw = nu * e[k] * x[k]
            self.w += dw
            nd[k,:] = dw * e[k]
            w_all[k:] = self.w
            self.w_history[k:] = self.w
        return y, e, self.w_history, nd





