"""
Classes of adaptive filters are stored in this file.
These classes can be used as standalone filters (e.g.real-time filtering).
"""
import numpy as np
import padasip.consts as co


class AdaptiveFilter():
    """
    Base class for adaptive filter classes. It puts together some functions
    used by all adaptive filters.
    """

    def init_weights(self, w, n):
        """
        This function initialises the adaptive weights of the filter.

        Args:

        * `w` : initial weights of filter. Possible values are:
        
            * array with initial weights (1 dimensional array) of filter size
        
            * "random" : create random weights
            
            * "zeros" : create zero value weights

        * `n` : size of filter (int) - number of filter coefficients.

        Returns:

        * `y` : output value (float) calculated from input array.

        """
        if type(w) == str:
            if w == "random":
                w = np.random.random(n)-0.5
            elif w == "zeros":
                w = np.zeros(n)
            else:
                raise ValueError('Impossible to understand the w')
        elif len(w) == n:
            try:
                w = np.array(w, dtype="float64")
            except:
                raise ValueError('Impossible to understand the w')
        else:
            raise ValueError('Impossible to understand the w')
        return w    

    def predict(self, x):
        """
        This function calculates the new output value `y` from input array `x`.

        Args:

        * `x` : input vector (1 dimension array) in length of filter.

        Returns:

        * `y` : output value (float) calculated from input array.

        """
        y = np.dot(self.w, x)
        return y

    def check_float_param(self, param, low, high, name):
        """
        Check if the value of the given parameter is in the given range
        and a float.
        Designed for testing parameters like `mu` and `eps`.
        To pass this function the variable `param` must be able to be converted
        into a float with a value between `low` and `high`.

        Args:

        * `param` : parameter to check (float or similar)

        * `low` : lowest allowed value (float), or None

        * `high` : highest allowed value (float), or None

        * `name` : name of the parameter (string), it is used for an error message
            
        Returns:

        * `param` : checked parameter converted to float

        """
        try:
            param = float(param)            
        except:
            raise ValueError(
                'Parameter {} is not float or similar'.format(name)
                )
        if low != None or high != None:
            if not low <= param <= high:
                raise ValueError('Parameter {} is not in range <{}, {}>'
                    .format(name, low, high))    
        return param 

    def check_int_param(self, param, low, high, name):
        """
        Check if the value of the given parameter is in the given range
        and an int.
        Designed for testing parameters like `mu` and `eps`.
        To pass this function the variable `param` must be able to be converted
        into a float with a value between `low` and `high`.

        Args:

        * `param` : parameter to check (int or similar)

        * `low` : lowest allowed value (int), or None

        * `high` : highest allowed value (int), or None

        * `name` : name of the parameter (string), it is used for an error message
            
        Returns:

        * `param` : checked parameter converted to float

        """
        try:
            param = int(param)            
        except:
            raise ValueError(
                'Parameter {} is not int or similar'.format(name)
                )
        if low != None or high != None:
            if not low <= param <= high:
                raise ValueError('Parameter {} is not in range <{}, {}>'
                    .format(name, low, high))    
        return param      



class FilterLMS(AdaptiveFilter):
    """
    This class represents an adaptive LMS filter.

    Args:

    * `n` : length of filter (integer) - how many input is input array
        (row of input matrix)

    Kwargs:

    * `mu` : learning rate (float). Also known as step size. If it is too slow,
        the filter may have bad performance. If it is too high,
        the filter will be unstable. The default value can be unstable
        for ill-conditioned input data.

    * `w` : initial weights of filter. Possible values are:
        
        * array with initial weights (1 dimensional array) of filter size
    
        * "random" : create random weights
        
        * "zeros" : create zero value weights
    """
    
    def __init__(self, n, mu=co.MU_LMS, w="random"):
        self.kind = "LMS filter"
        if type(n) == int:
            self.n = n
        else:
            raise ValueError('The size of filter must be an integer') 
        self.mu = self.check_float_param(mu, co.MU_LMS_MIN, co.MU_LMS_MAX, "mu")
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
        self.w += self.mu * e * x        

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
            dw = self.mu * e[k] * x[k]
            self.w += dw
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
            dw = self.mu * e[k] * x[k]
            self.w += dw
            nd[k,:] = dw * e[k]
            self.w_history[k:] = self.w
        return y, e, self.w_history, nd



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



class FilterRLS(AdaptiveFilter):
    """
    Adaptive RLS filter.

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
    def __init__(self, n, mu=co.MU_OCNLMS, eps=co.EPS_OCNLMS, 
            w="random", mem=co.MEM_OCNLMS):
        self.kind = "NLMS filter"
        if type(n) == int:
            self.n = n
        else:
            raise ValueError('The size of filter must be an integer')
        self.mu = self.check_float_param(mu, co.MU_OCNLMS_MIN, co.MU_OCNLMS_MAX, "mu")
        self.eps = self.check_float_param(eps, co.EPS_OCNLMS_MIN,
            co.EPS_OCNLMS_MAX, "eps")
        self.mem = self.check_int_param(mem, None, None, "mem")
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
            self.w_history[k:] = self.w
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
            self.w_history[k:] = self.w
            nd[k,:] = dw * e[k]
            self.update_memory_d(d[k])
        return y, e, self.w_history, nd


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





