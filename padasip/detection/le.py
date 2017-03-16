"""
.. versionadded:: 1.0.0

The Learning Entropy (LE) is non-Shannon entropy based on conformity
of individual data samples to the contemporary learned governing law
of a leraning system :cite:`bukovsky2013learning`. More information about
application can be found also in other studies :cite:`bukovsky2016study`
:cite:`bukovsky2015case` :cite:`bukovsky2014learning`.

Content of this page:

.. contents::
   :local:
   :depth: 1

Algorithm Explanation
==========================

Two options how to estimate the LE are implemented - direct approach and
multiscale approach.

.. rubric:: Direct approach

With direct approach the LE is evaluated for every sample as follows

:math:`\\textrm{LE}_d(k) = \\frac{ (\\Delta \\textbf{w}(k) - \\overline{| \\Delta \\textbf{w}_M(k) |}) }
{ (\\sigma({| \\Delta \\textbf{w}_M(k) |})+\\epsilon) }`

where

* :math:`|\\Delta \\textbf{w}(k)|` are the absolute values of current weights
  increment.

* :math:`\overline{| \\Delta \\textbf{w}_M(k) |}` are averages of absolute
  values of window used for LE evaluation.
  
* :math:`\\sigma (| \\Delta \\textbf{w}_M(k) |)` are standard deviatons of
  absolute values of window used for LE evaluation.

* :math:`\\epsilon` is regularization term to preserve stability for small
  values of standard deviation.


.. rubric:: Multiscale approach

Value for every sample is defined as follows

:math:`\\textrm{LE}(k) = \\frac{1}{n \cdot n_\\alpha}
\sum f(\Delta w_{i}(k), \\alpha ),`

where :math:`\Delta w_i(k)` stands for one weight from vector
:math:`\Delta \\textbf{w}(k)`, the :math:`n` is number of weights,
the :math:`n_\\alpha` is number of used detection sensitivities

:math:`\\alpha=[\\alpha_{1}, \\alpha_{2}, \ldots, \\alpha_{n_{\\alpha}}].`

The function :math:`f(\Delta w_{i}(k), \\alpha)` is defined as follows

:math:`f(\Delta w_{i}(k),\\alpha)=
\{{\\rm if}\,\left(\left\\vert \Delta w_{i}(k)\\right\\vert >
\\alpha\cdot \overline{\left\\vert \Delta w_{Mi}(k)\\right\\vert }\\right)\,
\\rm{then} \, 1, \\rm{else  }\,0 \}.`




Usage Instructions and Optimal Performance
==============================================

The LE algorithm can be used as follows

.. code-block:: python

    le = pa.detection.learning_entropy(w, m=30, order=1)
    
in case of direct approach. For multiscale approach an example follows

.. code-block:: python

    le = pa.detection.learning_entropy(w, m=30, order=1, alpha=[8., 9., 10., 11., 12., 13.])

where `w` is matrix of the adaptive parameters (changing in time, every row
should represent one time index), `m` is window size, `order` is LE order  and
`alpha` is vector of sensitivities.

.. rubric:: Used adaptive models

In general it is possible to use any adaptive model. The input of the LE
algorithm is matrix of an adaptive parameters history, where every row
represents the parameters used in a particular time and every column represents
one parameter in whole adaptation history.

.. rubric:: Selection of sensitivities

The optimal number of detection sensitivities and their values depends on task and data. The sensitivities should be chosen in range where the function :math:`LE(k)` returns a value lower than 1 for at least one sample in data, and for at maximally one sample returns value of 0.

Minimal Working Example
============================

In this example is demonstrated how can the multiscale approach LE highligh
the position of a perturbation inserted in a data. As the adaptive model is used 
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
    le = pa.detection.learning_entropy(w, m=30, order=2, alpha=[8., 9., 10., 11., 12., 13.])

    # LE plot
    plt.plot(le)
    plt.show()


References
============

.. bibliography:: le.bib
    :style: plain

Code Explanation
====================

"""
import numpy as np


def learning_entropy(w, m=10, order=1, alpha=False):
    """
    This function estimates Learning Entropy.

    **Args:**

    * `w` : history of adaptive parameters of an adaptive model (2d array),
      every row represents parameters in given time index.

    **Kwargs:**

    * `m` : window size (1d array) - how many last samples are used for
      evaluation of every sample.   
      
    * `order` : order of the LE (int) - order of weights differention

    * `alpha` : list of senstitivites (1d array). If not provided, the LE 
      direct approach is used.

    **Returns:**

    * Learning Entropy of data (1 d array) - one value for every sample

    """
    w = np.array(w)
    # get length of data and number of parameters
    N = w.shape[0]
    n = w.shape[1]
    # get abs dw from w
    dw = np.copy(w)
    dw[order:] = np.abs(np.diff(dw, n=order, axis=0))
    # average floting window - window is k-m ... k-1
    awd = np.zeros(w.shape)
    if not alpha:
        # estimate the ALPHA with multiscale approach
        swd = np.zeros(w.shape)
        for k in range(m, N):
            awd[k] = np.mean(dw[k-m:k], axis=0)
            swd[k] = np.std(dw[k-m:k], axis=0)
        # estimate the points of entropy
        eps = 1e-10 # regularization term
        le  = (dw - awd) / (swd+eps)
    else:
        # estimate the ALPHA with direct approach
        for k in range(m, N):
            awd[k] = np.mean(dw[k-m:k], axis=0)
        # estimate the points of entropy
        alphas = np.array(alpha)
        fh = np.zeros(N)
        for alpha in alphas:
            fh += np.sum(awd*alpha < dw, axis=1)
        le = fh / float(n*len(alphas))
    # clear unknown zone on begining
    le[:m] = 0
    # return output
    return le





