"""
This sub-module contains functions usefull for data preprocessing.
"""
from __future__ import division
import numpy as np

### data normalization
def standardize(x, offset=None, scale=None):
    """
    This function standardizes (z-score) the series according to equation
    
    :math:`\\textbf{x}_s = \\frac{\\textbf{x} - a}{b}`
    
    where :math:`\\textbf{x}` is time series to standardize,
    :math:`a` is offset to remove and :math:`b` scale to remove
    
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
        x = np.array(x, dtype="float64")
    except:
        raise ValueError('The argument x is not numpy array or similar.')         
    return (x - offset) / scale
    

def standardize_back(xs, offset, scale):
    """
    This function transforms series to the original score according to
    equation:

    :math:`\\textbf{x} = \\textbf{x}_s \cdot b + a`
    
    where :math:`\\textbf{x}` is time series to de-standardize,
    :math:`a` is offset to add and :math:`b` desired scaling factor.

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
