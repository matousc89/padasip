"""
.. versionadded:: 0.1

The normalized lest-mean-squares (NLMS) adaptive filter
:cite:`mandic2004generalized`
is an extension of the popular LMS adaptive filter (:ref:`filter-lms-label`).

The NLMS filter can be created as follows

    >>> import padasip as pa
    >>> pa.filters.FilterNLMS(n)
    
where `n` is the size (number of taps) of the filter.

Content of this page:

.. contents::
   :local:
   :depth: 1

Algorithm Explanation
*********************** 

The NLMS is extension of LMS filter. See :ref:`filter-lms-label`
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
**********************************

The stability of the NLMS filter si given as follows

:math:`0 \le \mu \le 2 + \\frac{2\epsilon}{||\\textbf{x}(k)||^2}`,

or in case without regularization term :math:`\epsilon`

:math:`\mu \in <0, 2>`.

In other words, if you use the zero or only small key argument `\eps`,
the key argument `\mu` should be between 0 and 2. Best convergence
should be produced by `mu=1.` according to theory. However in practice
the optimal value can be strongly case specific.

References
***************

.. bibliography:: nlms.bib
    :style: plain

Code Explanation
***************** 
"""
import numpy as np
import padasip.consts as co

from padasip.filters.base_filter import AdaptiveFilter



class FilterNLMS(AdaptiveFilter):
    """
    Adaptive NLMS filter.

    Args:

    * `n` : length of filter (integer) - how many input is input array
        (row of input matrix)

    Kwargs:

    * `mu` : learning rate (float). Also known as step size.
        If it is too slow,
        the filter may have bad performance. If it is too high,
        the filter will be unstable. The default value can be unstable
        for ill-conditioned input data.

    * `eps` : regularization term (float). It is introduced to preserve
        stability for close-to-zero input vectors

    * `w` : initial weights of filter. Possible values are:
        
        * array with initial weights (1 dimensional array) of filter size
    
        * "random" : create random weights
        
        * "zeros" : create zero value weights
    """ 
    def __init__(self, n, mu=co.MU_NLMS, eps=co.EPS_NLMS, w="random"):
        self.kind = "NLMS filter"
        if type(n) == int:
            self.n = n
        else:
            raise ValueError('The size of filter must be an integer') 
        self.mu = self.check_float_param(mu, co.MU_NLMS_MIN, co.MU_NLMS_MAX, "mu")
        self.eps = self.check_float_param(eps, co.EPS_NLMS_MIN,
            co.EPS_NLMS_MAX, "eps")
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
        nu = self.mu / (self.eps + np.dot(x, x))
        self.w += nu * e * x        

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
        self.w_history = np.zeros((N,self.n))
        # adaptation loop
        for k in range(N):
            y[k] = np.dot(self.w, x[k])
            e[k] = d[k] - y[k]
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
        self.w_history = np.zeros((N,self.n))
        # adaptation loop
        for k in range(N):
            y[k] = np.dot(self.w, x[k])
            e[k] = d[k] - y[k]
            nu = self.mu / (self.eps + np.dot(x[k], x[k]))
            dw = nu * e[k] * x[k]
            self.w += dw
            nd[k,:] = dw * e[k]
            self.w_history[k:] = self.w
        return y, e, self.w_history, nd
