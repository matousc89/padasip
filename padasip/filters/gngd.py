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
   

Minimal Working Examples
**************************

If you have measured data you may filter it as follows

.. code-block:: python

    # creation of data
    N = 500
    x = np.random.normal(0, 1, (N, 4)) # input matrix
    v = np.random.normal(0, 0.1, N) # noise
    d = 2*x[:,0] + 0.1*x[:,1] - 4*x[:,2] + 0.5*x[:,3] + v # target

    # identification
    f = pa.filters.FilterGNGD(n=4, mu=0.1, w="random")
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
    filt = pa.filters.FilterGNGD(3, mu=1.)
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
    **Args:**

    * `n` : length of filter (integer) - how many input is input array
        (row of input matrix)

    **Kwargs:**

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

        **Args:**

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

        **Args:**

        * `d` : desired value (1 dimensional array)

        * `x` : input matrix (2-dimensional array). Rows are samples, columns are
            input arrays.

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
            self.w_history[k,:] = self.w
        return y, e, self.w
        
    def novelty(self, d, x):
        """
        This function estimates novelty in data
        according to the learning effort.

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
            self.w_history[k,:] = self.w
        return y, e, self.w_history, nd





