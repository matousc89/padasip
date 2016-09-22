"""
.. versionadded:: 0.1

The RLS filter can be created as follows

    >>> import padasip as pa
    >>> pa.filters.FilterRLS(n)
    
where the n is amount of filter inputs (size of input vector).

Code Explanation
***************** 
"""

import numpy as np
import padasip.consts as co

from padasip.filters.base_filter import AdaptiveFilter

class FilterRLS(AdaptiveFilter):
    """
    The RLS filter class.

    Args:

    * `n` : length of filter (integer) - how many input is input array
        (row of input matrix)

    Kwargs:

    * `mu` : forgetting factor (float). It is introduced to give exponentially
        less weight to older error samples. It is usually chosen
        between 0.98 and 1.

    * `eps` : initialisation value (float). It is usually chosen
        between 0.1 and 1.

    * `w` : initial weights of filter. Possible values are:
        
        * array with initial weights (1 dimensional array) of filter size
    
        * "random" : create random weights
        
        * "zeros" : create zero value weights
    """ 

    def __init__(self, n, mu=co.MU_RLS, eps=co.EPS_RLS, w="random"):
        self.kind = "RLS filter"
        if type(n) == int:
            self.n = n
        else:
            raise ValueError('The size of filter must be an integer') 
        self.mu = self.check_float_param(mu, co.MU_RLS_MIN, co.MU_RLS_MAX, "mu")
        self.eps = self.check_float_param(eps, co.EPS_RLS_MIN, co.EPS_RLS_MAX, "eps")
        self.w = self.init_weights(w, self.n)
        self.R = 1/self.eps * np.identity(n)
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
        R1 = np.dot(np.dot(np.dot(self.R,x),x.T),self.R)
        R2 = self.mu + np.dot(np.dot(x,self.R),x.T)
        self.R = 1/self.mu * (self.R - R1/R2)
        dw = np.dot(self.R, x.T) * e
        self.w += dw

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
            R1 = np.dot(np.dot(np.dot(self.R,x[k]),x[k].T),self.R)
            R2 = self.mu + np.dot(np.dot(x[k],self.R),x[k].T)
            self.R = 1/self.mu * (self.R - R1/R2)
            dw = np.dot(self.R, x[k].T) * e[k]
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
        self.w_history = np.zeros((N,self.n))
        # adaptation loop
        for k in range(N):
            y[k] = np.dot(self.w, x[k])
            e[k] = d[k] - y[k]
            R1 = np.dot(np.dot(np.dot(self.R,x[k]),x[k].T),self.R)
            R2 = self.mu + np.dot(np.dot(x[k],self.R),x[k].T)
            self.R = 1/self.mu * (self.R - R1/R2)
            dw = np.dot(self.R, x[k].T) * e[k]
            self.w += dw
            nd[k,:] = dw * e[k]
            self.w_history[k:] = self.w
        return y, e, self.w_history, nd

