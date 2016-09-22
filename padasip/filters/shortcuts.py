from __future__ import division
import numpy as np
from padasip.filters import FilterLMS, FilterNLMS, FilterOCNLMS
from padasip.filters import FilterRLS, FilterGNGD
import padasip.consts as co

### LMS
def lms_filter(d, x, mu=co.MU_LMS, w="random"):
    """
    Process data with LMS filter.

    Args:

    * `d` : desired value (1 dimensional array)

    * `x` : input matrix (2-dimensional array). Rows are samples, columns are
        input arrays.

    Kwargs:

    * `mu` : learning rate (float). Also known as step size. If it is too slow,
        the filter may have bad performance. If it is too high,
        the filter will be unstable. The default value can be unstable
        for ill-conditioned input data.

    * `w` : initial weights of filter. Possible values are:
        
        * array with initial weights (1 dimensional array) of filter size
    
        * "random" : create random weights
        
        * "zeros" : create zero value weights
        
    Returns:

    * `y` : output value (1 dimensional array).
        The size corresponds with the desired value.

    * `e` : filter error for every sample (1 dimensional array). 
        The size corresponds with the desired value.

    * `w` : set of weights at the end of filtering (1 dimensional array). The size
        corresponds with the filter size (length of input vector).

    Example:

        >>> import numpy as np
        >>> import padasip as pa
        >>> x = np.random.random((10, 3))
        >>> d = np.sum(x, axis=1) # d = x1(k) + x2(k) + x3(k)
        >>> y, e, w = pa.lms_filter(d, x, mu=1.)
        >>> w
        array([ 1.0000086 ,  0.99996222,  1.00004785])


    """
    n = len(x[0])
    f = FilterLMS(n, mu, w)
    y, e, w = f.run(d, x)
    return y, e, w

def lms_novelty(d, x, mu=co.MU_LMS, w="random"):
    """
    Evaluate novelty in data with LMS filter. 

    Args:

    * `d` : desired value (1 dimensional array)

    * `x` : input matrix (2-dimensional array). Rows are samples, columns are
        input arrays.

    Kwargs:

    * `mu` : learning rate (float). Also known as step size. If it is too slow,
        filter will perform badly. If it is too high,
        filter will be unstable. The default value can be unstable
        for ill conditioned input data.

    * `w` : initial weights of filter. Possible values are:
        
        * array with initial weights (1 dimensional array) of filter size
    
        * "random" : create random weights
        
        * "zeros" : create zero value weights
        
    Returns:

    * `y` : output value (1 dimensional array).
        The size correspons with desired value.

    * `e` : filter error for every sample (1 dimensional array). 
        The size correspons with desired value.

    * `w` : history of all filter weighs (2 dimensional array).
        The size corresponds with lenght of data
        and filter size (length of input vector).

    * `nd` : coeficients describing novelty in data (2 dimensional array).
        The size corresponds with lenght of data
        and filter size (length of input vector).

    """
    n = len(x[0])
    f = FilterLMS(n, mu, w)
    y, e, w, nd = f.novelty(d, x)
    return y, e, w, nd


### NLMS
def nlms_filter(d, x, mu=co.MU_NLMS, eps=co.EPS_NLMS, w="random"):
    """
    Process data with NLMS filter.

    Args:

    * `d` : desired value (1 dimensional array)

    * `x` : input matrix (2-dimensional array). Rows are samples, columns are
        input arrays.

    Kwargs:

    * `mu` : learning rate (float). Also known as step size. If it is too slow,
        the filter may have bad performance. If it is too high,
        the filter will be unstable. The default value can be unstable
        for ill-conditioned input data.

    * `eps` : regularization term (float). It is introduced to preserve
        stability for close-to-zero input vectors

    * `w` : initial weights of filter. Possible values are:
        
        * array with initial weights (1 dimensional array) of filter size
    
        * "random" : create random weights
        
        * "zeros" : create zero value weights
        
    Returns:

    * `y` : output value (1 dimensional array).
        The size corresponds with the desired value.

    * `e` : filter error for every sample (1 dimensional array). 
        The size corresponds with the desired value.

    * `w` : set of weights at the end of filtering (1 dimensional array). The size
        corresponds with the filter size (length of input vector).
    """
    n = len(x[0])
    f = FilterNLMS(n, mu, eps, w)
    y, e, w = f.run(d, x)
    return y, e, w

def nlms_novelty(d, x, mu=co.MU_NLMS, eps=co.EPS_NLMS, w="random"):
    """
    Evaluate novelty in data with NLMS filter. 

    Args:

    * `d` : desired value (1 dimensional array)

    * `x` : input matrix (2-dimensional array). Rows are samples, columns are
        input arrays.

    Kwargs:

    * `mu` : learning rate (float). Also known as step size. If it is too slow,
        filter will perform badly. If it is too high,
        filter will be unstable. The default value can be unstable
        for ill conditioned input data.

    * `eps` : regularization term (float). It is introduced to preserve
        stability for close-to-zero input vectors

    * `w` : initial weights of filter. Possible values are:
        
        * array with initial weights (1 dimensional array) of filter size
    
        * "random" : create random weights
        
        * "zeros" : create zero value weights
        
    Returns:

    * `y` : output value (1 dimensional array).
        The size correspons with desired value.

    * `e` : filter error for every sample (1 dimensional array). 
        The size correspons with desired value.

    * `w` : history of all filter weighs (2 dimensional array).
        The size corresponds with lenght of data
        and filter size (length of input vector).

    * `nd` : coeficients describing novelty in data (2 dimensional array).
        The size corresponds with lenght of data
        and filter size (length of input vector).

    """
    n = len(x[0])
    f = FilterNLMS(n, mu, eps, w)
    y, e, w, nd = f.novelty(d, x)
    return y, e, w, nd

### OCNLMS
def ocnlms_filter(d, x, mu=co.MU_OCNLMS, eps=co.EPS_OCNLMS, w="random",
        mem=co.MEM_OCNLMS):
    """
    Process data with OC-NLMS filter.

    Args:

    * `d` : desired value (1 dimensional array)

    * `x` : input matrix (2-dimensional array). Rows are samples, columns are
        input arrays.

    Kwargs:

    * `mu` : learning rate (float). Also known as step size. If it is too slow,
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
        
    Returns:

    * `y` : output value (1 dimensional array).
        The size corresponds with the desired value.

    * `e` : filter error for every sample (1 dimensional array). 
        The size corresponds with the desired value.

    * `w` : set of weights at the end of filtering (1 dimensional array). The size
        corresponds with the filter size (length of input vector).
    """
    n = len(x[0])
    f = FilterOCNLMS(n, mu, eps, w, mem)
    y, e, w = f.run(d, x)
    return y, e, w

def ocnlms_novelty(d, x, mu=co.MU_NLMS, eps=co.EPS_NLMS, w="random",
        mem=co.MEM_OCNLMS):
    """
    Evaluate novelty in data with OC-NLMS filter. 

    Args:

    * `d` : desired value (1 dimensional array)

    * `x` : input matrix (2-dimensional array). Rows are samples, columns are
        input arrays.

    Kwargs:

    * `mu` : learning rate (float). Also known as step size. If it is too slow,
        filter will perform badly. If it is too high,
        filter will be unstable. The default value can be unstable
        for ill conditioned input data.

    * `eps` : regularization term (float). It is introduced to preserve
        stability for close-to-zero input vectors

    * `w` : initial weights of filter. Possible values are:
        
        * array with initial weights (1 dimensional array) of filter size
    
        * "random" : create random weights
        
        * "zeros" : create zero value weights

    * `mem` : size of filter memory (int). This means how many last targets
        and input vectors will be used for centering of current input vector
        and target.
        
    Returns:

    * `y` : output value (1 dimensional array).
        The size correspons with desired value.

    * `e` : filter error for every sample (1 dimensional array). 
        The size correspons with desired value.

    * `w` : history of all filter weighs (2 dimensional array).
        The size corresponds with lenght of data
        and filter size (length of input vector).

    * `nd` : coeficients describing novelty in data (2 dimensional array).
        The size corresponds with lenght of data
        and filter size (length of input vector).

    """
    n = len(x[0])
    f = FilterOCNLMS(n, mu, eps, w, mem)
    y, e, w, nd = f.novelty(d, x)
    return y, e, w, nd

### RLS
def rls_filter(d, x, mu=co.MU_RLS, eps=co.EPS_RLS, w="zeros"):
    """
    Process data with RLS filter.

    Args:

    * `d` : desired value (1 dimensional array)

    * `x` : input matrix (2-dimensional array). Rows are samples, columns are
        input arrays.

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
        
    Returns:

    * `y` : output value (1 dimensional array).
        The size corresponds with the desired value.

    * `e` : filter error for every sample (1 dimensional array). 
        The size corresponds with the desired value.

    * `w` : set of weights at the end of filtering (1 dimensional array). The size
        corresponds with the filter size (length of input vector).
    """
    n = len(x[0])
    f = FilterRLS(n, mu, eps, w)
    y, e, w = f.run(d, x)
    return y, e, w

def rls_novelty(d, x, mu=co.MU_RLS, eps=co.EPS_RLS, w="zeros"):
    """
    Evaluate novelty in data with RLS filter. 

    Args:

    * `d` : desired value (1 dimensional array)

    * `x` : input matrix (2-dimensional array). Rows are samples, columns are
        input arrays.

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
        
    Returns:

    * `y` : output value (1 dimensional array).
        The size correspons with desired value.

    * `e` : filter error for every sample (1 dimensional array). 
        The size correspons with desired value.

    * `w` : history of all filter weighs (2 dimensional array).
        The size corresponds with lenght of data
        and filter size (length of input vector).

    * `nd` : coeficients describing novelty in data (2 dimensional array).
        The size corresponds with lenght of data
        and filter size (length of input vector).

    """
    n = len(x[0])
    f = FilterRLS(n, mu, eps, w)
    y, e, w, nd = f.novelty(d, x)
    return y, e, w, nd



### GNGD
def gngd_filter(d, x, mu=co.MU_GNGD, eps=co.EPS_GNGD,
    ro=co.RO_GNGD, w="random"):
    """
    Process data with GNGD filter.

    Args:

    * `d` : desired value (1 dimensional array)

    * `x` : input matrix (2-dimensional array). Rows are samples, columns are
        input arrays.

    Kwargs:

    * `mu` : learning rate (float). Also known as step size. If it is too slow,
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
        
    Returns:

    * `y` : output value (1 dimensional array).
        The size corresponds with the desired value.

    * `e` : filter error for every sample (1 dimensional array). 
        The size corresponds with the desired value.

    * `w` : set of weights at the end of filtering (1 dimensional array). The size
        corresponds with the filter size (length of input vector).
    """
    n = len(x[0])
    f = FilterGNGD(n, mu, eps, ro, w)
    y, e, w = f.run(d, x)
    return y, e, w

def gngd_novelty(d, x, mu=co.MU_GNGD, eps=co.EPS_GNGD,
    ro=co.RO_GNGD, w="random"):
    """
    Evaluate novelty in data with GNGD filter. 

    Args:

    * `d` : desired value (1 dimensional array)

    * `x` : input matrix (2-dimensional array). Rows are samples, columns are
        input arrays.

    Kwargs:

    * `mu` : learning rate (float). Also known as step size. If it is too slow,
        filter will perform badly. If it is too high,
        filter will be unstable. The default value can be unstable
        for ill conditioned input data.

    * `eps` : compensation term (float) at the beginning. It is adaptive
        parameter.

    * `ro` : step size adaptation parameter (float) at the beginning.
        It is adaptive parameter.

    * `w` : initial weights of filter. Possible values are:
        
        * array with initial weights (1 dimensional array) of filter size
    
        * "random" : create random weights
        
        * "zeros" : create zero value weights
        
    Returns:

    * `y` : output value (1 dimensional array).
        The size correspons with desired value.

    * `e` : filter error for every sample (1 dimensional array). 
        The size correspons with desired value.

    * `w` : history of all filter weighs (2 dimensional array).
        The size corresponds with lenght of data
        and filter size (length of input vector).

    * `nd` : coeficients describing novelty in data (2 dimensional array).
        The size corresponds with lenght of data
        and filter size (length of input vector).

    """
    n = len(x[0])
    f = FilterGNGD(n, mu, eps, ro, w)
    y, e, w, nd = f.novelty(d, x)
    return y, e, w, nd
