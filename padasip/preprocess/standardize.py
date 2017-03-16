"""
.. versionadded:: 0.1

This function standardizes (z-score) the series according to equation

:math:`\\textbf{x}_s = \\frac{\\textbf{x} - a}{b}`

where :math:`\\textbf{x}` is time series to standardize,
:math:`a` is offset to remove and :math:`b` scale to remove

.. contents::
   :local:
   :depth: 1

See also: :ref:`preprocess-standardize_back`

Usage Explanation
********************

As simple as

.. code-block:: python

    xs = pa.standardize(x, offset=a , scale=b)

If the key arguments :code:`offset` and :code:`scale` are not provided
(example below) the mean value and standard deviation of `x` is used. 

.. code-block:: python

    xs = pa.standardize(x)


 
Minimal Working Example
**************************

An example how to standarize (z-score) data:

.. code-block:: python

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


Code Explanation
***************** 
"""
from __future__ import division
import numpy as np

def standardize(x, offset=None, scale=None):
    """   
    This is function for standarization of input series.

    **Args:**

    * `x` : series (1 dimensional array)

    **Kwargs:**

    * `offset` : offset to remove (float). If not given, \
        the mean value of `x` is used.

    * `scale` : scale (float). If not given, \
        the standard deviation of `x` is used.
        
    **Returns:**

    * `xs` : standardized series
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
