"""
.. versionadded:: 0.7

Implemented functions:

* **Mean absolute error** (MAE, also  known as MAD - mean absolute deviation)

  :math:`\\textrm{MAE}=\\frac{1}{n} \sum _{i=1}^{n}(e_{i})`.

* **Mean squared error** (MSE, also known as MSD)

  :math:`\\textrm{MSE}=\\frac{1}{n} \sum _{i=1}^{n}(e_{i})^{2}`.

* **Root-mean-square error** (RMSE, also known as RMSD)

  :math:`\\textrm{RMSE} = \\sqrt{\\textrm{MSE}}`.

* **Logarithmic squared error** (returns a vector of values in dB!)

  :math:`\\textbf{logSE} = 10 \log_{10} (\\textbf{e}^{2})`

all functions are often used for evaluation of an error rather than just
the error itself or its mean value.

Usage instructions
====================

For MAE evaluation from two time series use

.. code-block:: python
    
    mse = pa.misc.MAE(x1, x2)

If you have the error already calculated, then just

.. code-block:: python

    mse = pa.misc.MAE(e)

The same instructions apply for the MSE, RMSE a logarithmic squared error

.. code-block:: python
    
    mse = pa.misc.MSE(x1, x2)
    rmse = pa.misc.RMSE(x1, x2)
    logse = pa.misc.logSE(x1, x2)

and from error

.. code-block:: python

    mse = pa.misc.MSE(e)
    rmse = pa.misc.RMSE(e)
    logse = pa.misc.logSE(e)


Minimal working examples
==========================

In the following example is estimated MSE for two series
(:code:`x1` and :code:`x1`):

.. code-block:: python

    import numpy as np
    import padasip as pa 

    x1 = np.array([1, 2, 3, 4, 5])
    x2 = np.array([5, 4, 3, 2, 1])
    mse = pa.misc.MSE(x1, x2)
    print(mse)

You can easily check that the printed result :code:`8.0` is correct MSE for
given series.

The following example displays, that you can use directly the error series
:code:`e` if you already have it.

.. code-block:: python

    import numpy as np
    import padasip as pa 

    # somewhere else in your project
    x1 = np.array([1, 2, 3, 4, 5])
    x2 = np.array([5, 4, 3, 2, 1])
    e = x1 - x2
    # you have just the error - e
    mse = pa.misc.MSE(e)
    print(mse)

Again, you can check the correctness of the answer easily.

Code Explanation
====================
"""
import numpy as np

def get_valid_error(x1, x2=-1):
    """
    Function that validates:

    * x1 is possible to convert to numpy array

    * x2 is possible to convert to numpy array (if exists)

    * x1 and x2 have the same length (if both exist)
    """
    # just error
    if type(x2) == int and x2 == -1:
        try:    
            e = np.array(x1)
        except:
            raise ValueError('Impossible to convert series to a numpy array')        
    # two series
    else:
        try:
            x1 = np.array(x1)
            x2 = np.array(x2)
        except:
            raise ValueError('Impossible to convert one of series to a numpy array')
        if not len(x1) == len(x2):
            raise ValueError('The length of both series must agree.')
        e = x1 - x2
    return e

def logSE(x1, x2=-1):
    """
    10 * log10(e**2)    
    This function accepts two series of data or directly
    one series with error.

    **Args:**

    * `x1` - first data series or error (1d array)

    **Kwargs:**

    * `x2` - second series (1d array) if first series was not error directly,\\
        then this should be the second series

    **Returns:**

    * `e` - logSE of error (1d array) obtained directly from `x1`, \\
        or as a difference of `x1` and `x2`. The values are in dB!

    """
    e = get_valid_error(x1, x2)
    return 10*np.log10(e**2)
    

def MAE(x1, x2=-1):
    """
    Mean absolute error - this function accepts two series of data or directly
    one series with error.

    **Args:**

    * `x1` - first data series or error (1d array)

    **Kwargs:**

    * `x2` - second series (1d array) if first series was not error directly,\\
        then this should be the second series

    **Returns:**

    * `e` - MAE of error (float) obtained directly from `x1`, \\
        or as a difference of `x1` and `x2`

    """
    e = get_valid_error(x1, x2)
    return np.sum(np.abs(e)) / float(len(e))

def MSE(x1, x2=-1):
    """
    Mean squared error - this function accepts two series of data or directly
    one series with error.

    **Args:**

    * `x1` - first data series or error (1d array)

    **Kwargs:**

    * `x2` - second series (1d array) if first series was not error directly,\\
        then this should be the second series

    **Returns:**

    * `e` - MSE of error (float) obtained directly from `x1`, \\
        or as a difference of `x1` and `x2`

    """
    e = get_valid_error(x1, x2)
    return np.dot(e, e) / float(len(e))

def RMSE(x1, x2=-1):
    """
    Root-mean-square error - this function accepts two series of data
    or directly one series with error.

    **Args:**

    * `x1` - first data series or error (1d array)

    **Kwargs:**

    * `x2` - second series (1d array) if first series was not error directly,\\
        then this should be the second series

    **Returns:**

    * `e` - RMSE of error (float) obtained directly from `x1`, \\
        or as a difference of `x1` and `x2`

    """
    e = get_valid_error(x1, x2)
    return np.sqrt(np.dot(e, e) / float(len(e)))

def get_mean_error(x1, x2=-1, function="MSE"):
    """
    This function returns desired mean error. Options are: MSE, MAE, RMSE
    
    **Args:**

    * `x1` - first data series or error (1d array)

    **Kwargs:**

    * `x2` - second series (1d array) if first series was not error directly,\\
        then this should be the second series

    **Returns:**

    * `e` - mean error value (float) obtained directly from `x1`, \\
        or as a difference of `x1` and `x2`
    """
    if function == "MSE":
        return MSE(x1, x2)
    elif function == "MAE":
        return MAE(x1, x2)
    elif function == "RMSE":
        return RMSE(x1, x2)
    else:
        raise ValueError('The provided error function is not known')

