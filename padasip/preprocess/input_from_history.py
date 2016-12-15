"""
.. versionadded:: 0.1

This function creates input matrix from historical values.

.. contents::
   :local:
   :depth: 1

Minimal Working Example
**************************

An example how to create input matrix from historical values

.. code-block:: python

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

Code Explanation
***************** 
"""
from __future__ import division
import numpy as np

def input_from_history(a, n, bias=False):
    """
    This is function for creation of input matrix.

    **Args:**

    * `a` : series (1 dimensional array)

    * `n` : size of input matrix row (int). It means how many samples \
        of previous history you want to use \
        as the filter input. It also represents the filter length.

    **Kwargs:**

    * `bias` : decides if the bias is used (Boolean). If True, \
        array of all ones is appended as a last column to matrix `x`. \
        So matrix `x` has `n`+1 columns.

    **Returns:**

    * `x` : input matrix (2 dimensional array) \
        constructed from an array `a`. The length of `x` \
        is calculated as length of `a` - `n` + 1. \
        If the `bias` is used, then the amount of columns is `n` if not then \
        amount of columns is `n`+1).

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
