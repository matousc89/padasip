"""
.. versionadded:: 0.1
.. versionchanged:: 0.7


An adaptive filter is a system that changes its adaptive parameteres
- adaptive weights :math:`\\textbf{w}(k)` - according to an optimization algorithm.

The an adaptive filter can be described as

:math:`y(k) = w_1 \cdot x_{1}(k) + ... + w_n \cdot x_{n}(k)`,

or in a vector form

:math:`y(k) = \\textbf{x}^T(k) \\textbf{w}(k)`.

The adaptation of adaptive parameters (weights) can be done with
various algorithms.

Content of this page:

.. contents::
   :local:
   :depth: 1

Usage instructions
================================================

.. rubric:: Adaptive weights initial selection

The parameters of all implemented adaptive filters can be initially set:

* manually and passed to a filter as an array

* :code:`w="random"` - set to random - this will produce a vector of
  random values (zero mean, 0.5 standard deviation)
    
* :code:`w="zeros"` - set to zeros

.. rubric:: Input data

The adaptive filters need two inputs

* input matrix :code:`x` where rows represent the samples. Every row (sample)
  should contain multiple values (features). 

* desired value (target) :code:`d` 

If you have only one signal and the historical values of this signal should
be input of the filter (data reconstruction/prediction task) you can use helper
function :ref:`preprocess-input_from_history` to build input matrix from
the historical values.

.. rubric:: Creation of an adaptive filter

If you want to create adaptive filter (for example NLMS), with size :code:`n=4`,
learning rate :code:`mu=0.1` and random initial parameters (weights), than use
following code

.. code-block:: python

    f = pa.filters.AdaptiveFilter(model="NLMS", n=4, mu=0.1, w="random")   

where returned :code:`f` is the instance of class :code:`FilterNLMS`
with given parameters.

.. rubric:: Data filtering

If you already created an instance of adaptive filter (:code:`f` in previous 
example) you can use it for filtering of 
data :code:`x` with desired value :code:`d` as simple as follows

.. code-block:: python  
    
    y, e, w = f.run(d, x)

where :code:`y` is output, :code:`e` is the error and :code:`w` is set
of parameters at the end of the simulation.

In case you want to just simply filter the data without creating and
storing filter instance manually, use the following function 

.. code-block:: python  
  
    y, e, w = pa.filters.filter_data(d, x, model="NLMS", mu=0.9, w="random")

    
.. rubric:: Search for optimal learning rate

The search for optimal filter setup (especially learning rate) is a task
of critical importance. Therefor an helper function for this task is
implemented in the Padasip. To use this function you need to specify

* number of epochs (for training)

* part of data used in training epochs - `ntrain` (0.5 stands for 50% of
  given data)
  
* start and end of learning rate range you want to test (and number of
  steps in this range) - `mu_start`, `mu_end`, `steps`
  
* testing criteria (MSE, RMSE, MAE)
  
Example for `mu` in range of 100 values from `[0.01, ..., 1]` follows.
In example is used 50% of data for training and leftoever data for testing
with MSE criteria. Returned arrays are list of errors and list of corresponding
learning rates, so it is easy to plot and analyze the error as
a function of learning rate.

.. code-block:: python  

    errors_e, mu_range = f.explore_learning(d, x,
                    mu_start=0.01,
                    mu_end=1.,
                    steps=100, ntrain=0.5, epochs=1,
                    criteria="MSE")
                    
Note: optimal learning rate depends on purpose and usage of filter (ammount
of training, data characteristics, etc.).
    

Full Working Example
===================================================

Bellow is full working example with visualisation of results - the NLMS
adaptive filter used for channel identification.

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
    f = pa.filters.AdaptiveFilter(model="NLMS", n=4, mu=0.1, w="random")
    y, e, w = f.run(d, x)

    ## show results
    plt.figure(figsize=(15,9))
    plt.subplot(211);plt.title("Adaptation");plt.xlabel("samples - k")
    plt.plot(d,"b", label="d - target")
    plt.plot(y,"g", label="y - output");plt.legend()
    plt.subplot(212);plt.title("Filter error");plt.xlabel("samples - k")
    plt.plot(10*np.log10(e**2),"r", label="e - error [dB]");plt.legend()
    plt.tight_layout()
    plt.show()


Implemented filters
========================

.. toctree::
    :glob:
    :maxdepth: 1

    filters/*

Code explanation
==================
"""
from padasip.filters.lms import FilterLMS
from padasip.filters.sslms import FilterSSLMS
from padasip.filters.lmf import FilterLMF
from padasip.filters.nlms import FilterNLMS
from padasip.filters.nsslms import FilterNSSLMS
from padasip.filters.nlmf import FilterNLMF
from padasip.filters.fxlms import FilterFxLMS
from padasip.filters.ocnlms import FilterOCNLMS
from padasip.filters.gngd import FilterGNGD
from padasip.filters.rls import FilterRLS
from padasip.filters.ap import FilterAP

def filter_data(d, x, model="lms", **kwargs):
    """
    Function that filter data with selected adaptive filter.
    
    **Args:**

    * `d` : desired value (1 dimensional array)

    * `x` : input matrix (2-dimensional array). Rows are samples, columns are
      input arrays.
        
    **Kwargs:**
    
    * Any key argument that can be accepted with selected filter model. 
      For more information see documentation of desired adaptive filter.

    **Returns:**

    * `y` : output value (1 dimensional array).
      The size corresponds with the desired value.

    * `e` : filter error for every sample (1 dimensional array). 
      The size corresponds with the desired value.

    * `w` : history of all weights (2 dimensional array).
      Every row is set of the weights for given sample.
    
    """
    # overwrite n with correct size
    kwargs["n"] = x.shape[1]
    # create filter according model
    if model in ["LMS", "lms"]:
        f = FilterLMS(**kwargs)
    elif model in ["NLMS", "nlms"]:
        f = FilterNLMS(**kwargs)
    elif model in ["RLS", "rls"]:
        f = FilterRLS(**kwargs)
    elif model in ["GNGD", "gngd"]:
        f = FilterGNGD(**kwargs)
    elif model in ["AP", "ap"]:
        f = FilterAP(**kwargs)
    elif model in ["LMF", "lmf"]:
        f = FilterLMF(**kwargs)
    elif model in ["NLMF", "nlmf"]:
        f = FilterNLMF(**kwargs)
    else:
        raise ValueError('Unknown model of filter {}'.format(model))
    # calculate and return the values
    y, e, w = f.run(d, x)
    return y, e, w

def AdaptiveFilter(model="lms", **kwargs):
    """
    Function that filter data with selected adaptive filter.
    
    **Args:**

    * `d` : desired value (1 dimensional array)

    * `x` : input matrix (2-dimensional array). Rows are samples, columns are 
      input arrays.
        
    **Kwargs:**
    
    * Any key argument that can be accepted with selected filter model.
      For more information see documentation of desired adaptive filter.
    
    * It should be at least filter size `n`.  

    **Returns:**

    * `y` : output value (1 dimensional array).
      The size corresponds with the desired value.

    * `e` : filter error for every sample (1 dimensional array). 
      The size corresponds with the desired value.

    * `w` : history of all weights (2 dimensional array).
      Every row is set of the weights for given sample.
    
    """
    # check if the filter size was specified
    if not "n" in kwargs:
        raise ValueError('Filter size is not defined (n=?).')    
    # create filter according model
    if model in ["LMS", "lms"]:
        f = FilterLMS(**kwargs)
    elif model in ["NLMS", "nlms"]:
        f = FilterNLMS(**kwargs)
    elif model in ["RLS", "rls"]:
        f = FilterRLS(**kwargs)
    elif model in ["GNGD", "gngd"]:
        f = FilterGNGD(**kwargs)
    elif model in ["AP", "ap"]:
        f = FilterAP(**kwargs)
    elif model in ["LMF", "lmf"]:
        f = FilterLMF(**kwargs)
    elif model in ["NLMF", "nlmf"]:
        f = FilterNLMF(**kwargs)
    else:
        raise ValueError('Unknown model of filter {}'.format(model))
    # return filter
    return f

