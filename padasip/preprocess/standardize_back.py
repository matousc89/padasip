"""
.. versionadded:: 0.1

This function transforms series to the original score according to
equation:

:math:`\\textbf{x} = \\textbf{x}_s \cdot b + a`

where :math:`\\textbf{x}` is time series to de-standardize,
:math:`a` is offset to add and :math:`b` desired scaling factor.

.. contents::
   :local:
   :depth: 1

See also: :ref:`preprocess-standardize`

Usage Explanation
********************

As simple as

.. code-block:: python

    x = pa.standardize(xs, offset=a, scale=b)

Code Explanation
***************** 
"""
from __future__ import division
import numpy as np

def standardize_back(xs, offset, scale):
    """
    This is function for de-standarization of input series.

    **Args:**

    * `xs` : standardized input (1 dimensional array)

    * `offset` : offset to add (float).

    * `scale` : scale (float).
        
    **Returns:**

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

