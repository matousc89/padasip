"""
.. versionchanged:: 1.1.0

The normalized sign-sign least-mean-squares (NSSLMS) adaptive filter
:cite:`rahman2009noise`
is an extension of the popular SSLMS adaptive filter (:ref:`filter-sslms`).

The NSSLMS filter can be created as follows

    >>> import padasip as pa
    >>> pa.filters.FilterNSSLMS(n)
    
where `n` is the size (number of taps) of the filter.

Content of this page:

.. contents::
   :local:
   :depth: 1

.. seealso:: :ref:`filters`

Algorithm Explanation
======================================

The NSSLMS is extension of LMS filter. See :ref:`filter-lms`
for explanation of the algorithm behind.

The extension is based on normalization of learning rate.
The learning rage :math:`\mu` is replaced by learning rate :math:`\eta(k)`
normalized with every new sample according to input power as follows

:math:`\eta (k) = \\frac{\mu}{\epsilon + || \\textbf{x}(k) ||^2}`,

where :math:`|| \\textbf{x}(k) ||^2` is norm of input vector and 
:math:`\epsilon` is a small positive constant (regularization term).
This constant is introduced to preserve the stability in cases where
the input is close to zero.

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
    d = 2*x[:,0] + 0.1*x[:,1] - 0.3*x[:,2] + 0.5*x[:,3] + v # target

    # identification
    f = pa.filters.FilterNSSLMS(n=4, mu=0.1, w="random")
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
======================================

.. bibliography:: sslms.bib
    :style: plain

Code Explanation
======================================
"""
import numpy as np

from padasip.filters.base_filter import AdaptiveFilter

class FilterNSSLMS(AdaptiveFilter):
    """
    Adaptive NSSLMS filter.

    **Args:**

    * `n` : length of filter (integer) - how many input is input array
      (row of input matrix)

    **Kwargs:**

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
    def __init__(self, n, mu=0.1, eps=1., w="random"):
        self.kind = "NSSLMS filter"
        if type(n) == int:
            self.n = n
        else:
            raise ValueError('The size of filter must be an integer') 
        self.mu = self.check_float_param(mu, 0, 1000, "mu")
        self.eps = self.check_float_param(eps, 0, 1000, "eps")
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
        nu = self.mu / (self.eps + np.dot(x, x))
        self.w += self.mu * np.sign(e) * np.sign(x)      

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
            nu = self.mu / (self.eps + np.dot(x[k], x[k]))
            dw = nu * np.sign(e[k]) * np.sign(x[k])
            self.w += dw
        return y, e, self.w_history
        
