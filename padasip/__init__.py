"""
This library is designed to simplify adaptive signal 
processing tasks within python
(filtering, prediction, reconstruction, classification).
For code optimisation, this library uses numpy for array operations.

Also in this library is presented some new methods
for adaptive signal processing.
The library is designed to be used with datasets and also with 
real-time measuring (sample-after-sample feeding).
"""
import numpy as np
from padasip.filters import FilterLMS, FilterNLMS, FilterRLS
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





### data normalization

def standardize(x, offset=None, scale=None):
    """
    This function standardizes the series.

    xs = (x - offset) / scale

    Args:

    * `x` : series (1 dimensional array)

    Kwargs:

    * `offset` : offset to remove (float). If not given,
        the mean value of `x` is used.

    * `scale` : scale (float). If not given,
        the standard deviation of `x` is used.
        
    Returns:

    * `xs` : standardized series

    Example:


        >>> import numpy as np
        >>> import padasip as pa
        >>> x = np.random.random(1000)
        >>> x.mean()
        0.49755420774866677
        >>> x.std()
        0.29015765297767376
        >>> xs = pa.standardize(x)
        >>> xs.mean()
        1.4123424652012772e-16
        >>> xs.std()
        0.99999999999999989

    """
    if offset == None:
        offset = np.array(x).mean()
    else:
        try:
            offset = float(offset)
        except:
            raise ValueError('The argument offset is not None or float') 
    if scale == None:
        scale = np.array(x).std()
    else:
        try:
            scale = float(scale)
        except:
            raise ValueError('The argument scale is not None or float')    
    try:
        xs = np.array(xs, dtype="float64")
    except:
        raise ValueError('The argument x is not numpy array or similar.')         
    return (x - offset) / scale
    

def standardize_back(xs, offset, scale):
    """
    This function transforms series to the original score.

    x = xs * scale + offset

    Args:

    * `xs` : standardized input (1 dimensional array)

    * `offset` : offset to add (float).

    * `scale` : scale (float).
        
    Returns:

    * `x` : original (destandardised) series
    """
    try:
        offset = float(offset)
    except:
        raise ValueError('The argument offset is not None or float.') 
    try:
        scale = float(scale)
    except:
        raise ValueError('The argument scale is not None or float.')
    try:
        xs = np.array(xs, dtype="float64")
    except:
        raise ValueError('The argument xs is not numpy array or similar.')
    return xs*scale + offset


### INPUT MATRIX constructors

def input_from_history(a, n, bias=False):
    """
    This function creates input matrix from historical values.

    Args:

    * `a` : series (1 dimensional array)

    * `n` : size of input matrix row (int).
        It means how many samples of previous history you want to use
        as the filter input. It also represents the filter length.

    Kwargs:

    * `bias` : decides if the bias is used (Boolean). If True,
        array of all ones is appended as a last column to matrix `x`.
        So matrix `x` has `n`+1 columns.

    Returns:

    * `x` : input matrix (2 dimensional array)
        constructed from an array `a`. The length of `x`
        is calculated as length of `a` - `n` + 1. 
        If the `bias` is used, then the amount of columns is `n` if not then
        amount of columns is `n`+1).

    Example:

        >>> import numpy as np
        >>> import padasip as pa
        >>> a = np.arange(1, 7, 1)
        >>> a
        array([1, 2, 3, 4, 5, 6])
        >>> pa.input_from_history(a,3)
        array([[1, 2, 3],
               [2, 3, 4],
               [3, 4, 5],
               [4, 5, 6]])

    """
    if not type(n) == int:
        raise ValueError('The argument n must be int.')
    if not n > 0:
        raise ValueError('The argument n must be greater than 0')
    try:
        a = np.array(a, dtype="float64")
    except:
        raise ValueError('The argument a is not numpy array or similar.')
    x = np.array([a[i:i+n] for i in range(len(a)-n+1)]) 
    if bias:
        x = np.vstack((x.T, np.ones(len(x)))).T
    return x












