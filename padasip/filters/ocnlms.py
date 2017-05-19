import numpy as np

from padasip.filters.base_filter import AdaptiveFilter

class FilterOCNLMS(AdaptiveFilter):
    """
    Adaptive OC-NLMS filter.

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

    * `mem` : size of filter memory (int). This means how many last targets
        and input vectors will be used for centering of current input vector
        and target.
    """
    def __init__(self, n, mu=0.1, eps=1., 
            w="random", mem=100):
        self.kind = "OC-NLMS filter"
        if type(n) == int:
            self.n = n
        else:
            raise ValueError('The size of filter must be an integer')
        self.mu = self.check_float_param(mu, 0, 1000, "mu")
        self.eps = self.check_float_param(eps, 0, 1000, "eps")
        self.mem = self.check_int_param(mem, 1, 1000, "mem")
        self.w = self.init_weights(w, self.n)
        self.w_history = False
        self.mem_empty = True
        self.mem_x = np.zeros((mem,n))
        self.mem_d = np.zeros(mem)
        self.mem_idx = 0

    def adapt(self, d, x):
        """
        Adapt weights according one desired value and its input.

        Args:

        * `d` : desired value (float)

        * `x` : input array (1-dimensional array)
        """
        self.update_memory_x(x)
        m_d, m_x = self.read_memory()
        # estimate
        y = np.dot(self.w, x-m_x) + m_d
        e = d - y
        nu = self.mu / (self.eps + np.dot(x-m_x, x-m_x))
        dw = nu * e * (x-m_x)
        self.w += dw
        self.update_memory_d(d)

    def update_memory_d(self, d_k):
        """
        This function update memory of the filter with new target value `d`.
        """
        self.mem_d[self.mem_idx-1] = d_k

    def update_memory_x(self, x_k):
        """
        This function update memory of the filter with new input vector `x`.
        """
        self.mem_x[self.mem_idx, :] = x_k

    def read_memory(self):
        """
        This function read mean value of target`d`
        and input vector `x` from history
        """
        if self.mem_empty == True:
            if self.mem_idx == 0:
                m_x = np.zeros(self.n)
                m_d = 0
            else:
                m_x = np.mean(self.mem_x[:self.mem_idx+1], axis=0)
                m_d = np.mean(self.mem_d[:self.mem_idx])
        else:
            m_x = np.mean(self.mem_x, axis=0)
            m_d = np.mean(np.delete(self.mem_d, self.mem_idx))
        self.mem_idx += 1
        if self.mem_idx > len(self.mem_x)-1:
            self.mem_idx = 0
            self.mem_empty = False
        return m_d, m_x

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
            self.update_memory_x(x[k])
            m_d, m_x = self.read_memory()
            # estimate
            y[k] = np.dot(self.w, x[k]-m_x) + m_d
            e[k] = d[k] - y[k]
            nu = self.mu / (self.eps + np.dot(x[k]-m_x, x[k]-m_x))
            dw = nu * e[k] * (x[k]-m_x)
            self.w += dw
            self.w_history[k,:] = self.w
            self.update_memory_d(d[k])
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
            self.update_memory_x(x[k])
            m_d, m_x = self.read_memory()
            # estimate
            y[k] = np.dot(self.w, x[k]-m_x) + m_d
            e[k] = d[k] - y[k]
            nu = self.mu / (self.eps + np.dot(x[k]-m_x, x[k]-m_x))
            dw = nu * e[k] * (x[k]-m_x)
            self.w += dw
            self.w_history[k,:] = self.w
            nd[k,:] = dw * e[k]
            self.update_memory_d(d[k])
        return y, e, self.w_history, nd

