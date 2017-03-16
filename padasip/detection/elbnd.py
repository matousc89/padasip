"""
.. versionadded:: 1.0.0

The Error and Learning Based Novelty Detection (ELBND)is based on the
evaluation of an adaptive model error and the change of its parameters
:cite:`cejnek2015adaptive`, :cite:`cejnek2014another`.

Content of this page:

.. contents::
   :local:
   :depth: 1

Algorithm Explanation
==========================

The ELBND can describe every sample with vector of values estimated from
the adaptive increments of any adaptive model and the error of that model
as follows

:math:`\\textrm{ELBND}(k) = \Delta \\textbf{w}(k) e(k).`

The output is a vector of values describing novelty in given sample.
To obtain single value of novelty ammount for every sample is possible to use
various functions, for example maximum of absolute values.

:math:`\\textrm{elbnd}(k) = \max |\\textrm{ELBND}(k)|.`

Other popular option is to make a sum of absolute values.


Usage Instructions
========================

The ELBND algorithm can be used as follows

.. code-block:: python

    elbnd = pa.detection.ELBND(w, e, function="max")

where `w` is matrix of the adaptive parameters (changing in time, every row
should represent one time index), `e` is error of adaptive model and
`function` is input function, in this case maximum. 


Minimal Working Example
============================

In this example is demonstrated how can the LE highligh the position of
a perturbation inserted in a data. As the adaptive model is used 
:ref:`filter-nlms` adaptive filter. The perturbation is manually inserted
in sample with index :math:`k=1000` (the length of data is 2000).

.. code-block:: python

    import numpy as np
    import matplotlib.pylab as plt
    import padasip as pa 

    # data creation
    n = 5
    N = 2000
    x = np.random.normal(0, 1, (N, n))
    d = np.sum(x, axis=1) + np.random.normal(0, 0.1, N)

    # perturbation insertion
    d[1000] += 2.

    # creation of learning model (adaptive filter)
    f = pa.filters.FilterNLMS(n, mu=1., w=np.ones(n))
    y, e, w = f.run(d, x)

    # estimation of LE with weights from learning model
    elbnd = pa.detection.ELBND(w, e, function="max")

    # LE plot
    plt.plot(elbnd)
    plt.show()


References
============

.. bibliography:: elbnd.bib
    :style: plain

Code Explanation
====================

"""
import numpy as np

def ELBND(w, e, function="max"):
    """
    This function estimates Error and Learning Based Novelty Detection measure
    from given data.

    **Args:**

    * `w` : history of adaptive parameters of an adaptive model (2d array),
      every row represents parameters in given time index.

    * `e` : error of adaptive model (1d array)

    **Kwargs:**

    * `functions` : output function (str). The way how to produce single
      value for every sample (from all parameters)
      
        * `max` - maximal value
      
        * `sum` - sum of values

    **Returns:**

    * ELBND values (1d array). This vector has same lenght as `w`.

    """
    # check if the function is known
    if not function in ["max", "sum"]:
        raise ValueError('Unknown output function')
    # get length of data and number of parameters
    N = w.shape[0]
    n = w.shape[1]
    # get abs dw from w
    dw = np.zeros(w.shape)
    dw[:-1] = np.abs(np.diff(w, axis=0))
    # absolute values of product of increments and error
    a = np.random.random((5,2))
    b = a.T*np.array([1,2,3,4,5])
    elbnd = np.abs((dw.T*e).T)
    # apply output function
    if function == "max":
        elbnd = np.max(elbnd, axis=1)
    elif function == "sum":
        elbnd = np.sum(elbnd, axis=1) 
    # return output
    return elbnd




