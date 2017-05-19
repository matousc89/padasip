"""
.. versionadded:: 0.4
.. versionchanged:: 1.0.0

The Affine Projection (AP) algorithm is implemented according to paper
:cite:`gonzalez2012affine`. Usage of this filter should be benefical especially
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
:math:`k` is discrete time index and \textbf{x}_{k} is input vector.
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
    f = pa.filters.FilterAP(n=4, order=5, mu=0.5, eps=0.001, w="random")
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


References
======================================

.. bibliography:: ap.bib
    :style: plain

Code Explanation
======================================
"""
import numpy as np

from padasip.filters.base_filter import AdaptiveFilter

class FilterAP(AdaptiveFilter):
    """
    Adaptive AP filter.
    
    **Args:**

    * `n` : length of filter (integer) - how many input is input array
      (row of input matrix)

    **Kwargs:**

    * `order` : projection order (integer) - how many input vectors
      are in one input matrix

    * `mu` : learning rate (float). Also known as step size.
      If it is too slow,
      the filter may have bad performance. If it is too high,
      the filter will be unstable. The default value can be unstable
      for ill-conditioned input data.

    * `eps` : initial offset covariance (float)
    
    * `w` : initial weights of filter. Possible values are:
        
        * array with initial weights (1 dimensional array) of filter size
    
        * "random" : create random weights
        
        * "zeros" : create zero value weights
    """ 
    def __init__(self, n, order=5, mu=0.1, eps=0.001, w="random"):
        self.kind = "AP filter"
        self.n = self.check_int(
            n,'The size of filter must be an integer')
        self.order = self.check_int(
            order, 'The order of projection must be an integer')
        self.mu = self.check_float_param(mu, 0, 1000, "mu")
        self.eps = self.check_float_param(eps, 0, 1000, "eps")
        self.init_weights(w, self.n)
        self.w_history = False
        self.x_mem = np.zeros((self.n, self.order))
        self.d_mem = np.zeros(order)
        self.ide_eps = self.eps * np.identity(self.order)
        self.ide = np.identity(self.order)
        self.y_mem = False
        self.e_mem = False

    def adapt(self, d, x):
        """
        Adapt weights according one desired value and its input.

        **Args:**

        * `d` : desired value (float)

        * `x` : input array (1-dimensional array)
        """
        # create input matrix and target vector
        self.x_mem[:,1:] = self.x_mem[:,:-1]
        self.x_mem[:,0] = x
        self.d_mem[1:] = self.d_mem[:-1]
        self.d_mem[0] = d
        # estimate output and error
        self.y_mem = np.dot(self.x_mem.T, self.w)
        self.e_mem = self.d_mem - self.y_mem
        # update
        dw_part1 = np.dot(self.x_mem.T, self.x_mem) + self.ide_eps
        dw_part2 = np.linalg.solve(dw_part1, self.ide)
        dw = np.dot(self.x_mem, np.dot(dw_part2, self.e_mem))
        self.w += self.mu * dw   

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
            # create input matrix and target vector
            self.x_mem[:,1:] = self.x_mem[:,:-1]
            self.x_mem[:,0] = x[k]
            self.d_mem[1:] = self.d_mem[:-1]
            self.d_mem[0] = d[k]
            # estimate output and error
            self.y_mem = np.dot(self.x_mem.T, self.w)
            self.e_mem = self.d_mem - self.y_mem
            y[k] = self.y_mem[0]
            e[k] = self.e_mem[0]
            # update
            dw_part1 = np.dot(self.x_mem.T, self.x_mem) + self.ide_eps
            dw_part2 = np.linalg.solve(dw_part1, self.ide)
            dw = np.dot(self.x_mem, np.dot(dw_part2, self.e_mem))
            self.w += self.mu * dw           
        return y, e, self.w_history
        
