"""
.. versionadded:: 1.2.0

The Extreme Seeking Entropy (ESE) introduced
in https://doi.org/10.3390/e22010093 is based on the
evaluation of a change of adaptive model parameters.
This function requires `SciPy <https://pypi.org/project/scipy/>`_.

Content of this page:

.. contents::
   :local:
   :depth: 1

Algorithm Explanation
==========================

The ESE can describe every sample with a value,
that is proportional impropability value of adaptive increments.
The probability of value of adaptive increment that is higher
than some threshold value is estimated from Generalized Pareto distribution.
Value of ESE.

Usage Instructions
========================

The ESE algorithm can be used as follows

.. code-block:: python

    ese = pa.detection.ELBND(w, window=1000)

where `w` is matrix of the adaptive parameters (changing in time, every row
should represent one time index) and
`window` is a size of the window used for distribution estimation.
The length of the provided data `w` has to greter than 'window'.
The first `window` number of samples cannot be evaluated with ESE.


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
    N = 5000
    x = np.random.normal(0, 1, (N, n))
    d = np.sum(x, axis=1) + np.random.normal(0, 0.1, N)

    # fake perturbation insertion
    d[2500] += 2.

    # creation of learning model (adaptive filter)
    f = pa.filters.FilterNLMS(n, mu=1., w=np.ones(n))
    y, e, w = f.run(d, x)

    # estimation of ESE with weights from learning model
    ese = pa.detection.ESE(w)

    # ese plot
    plt.plot(ese)
    plt.show()

Code Explanation
====================

"""
import numpy as np
from scipy.stats import genpareto

def pot(data, method):
    """
    Peak-Over-Threshold method.
    :param data: input data (n samples)
    :param method: method identifier
    :return: k highest values
    """
    sorted_data = -np.sort(-data)
    k = 0
    n = len(data)
    if method == "10%":
        k = max(int(0.1 * n), 1)
    elif method == "sqrt":
        k = max(int(np.sqrt(n)), 1)
    elif method == "log10log10":
        k = max(int((n ** (2/3))/np.log10(np.log10(n))), 1)
    elif method == "log10":
        k = max(int(np.log10(n)), 1)
    elif method == "35%":
        k = max(int(0.35 * n), 1)
    return sorted_data[:k]

def ESE(w, window=1000, pot_method="10%"):
    """
    This function estimates Extreme Seeking Entropy measure
    from given data.

    **Args:**

    * `w` : history of adaptive parameters of an adaptive model (2d array),
      every row represents parameters in given time index.

    **Kwargs:**

    * `window` : number of samples that are proceeded via P-O-T method

    * `pot_method` : identifier of P-O-T method (str): 'sqrt', '10%', '30%', 'log10', 'log10log10'

    **Returns:**

    *  values of Extreme Seeking Entropy (1d array). This vector has same lenght as `w`.

    """
    filter_len = w.shape[1]
    dw = np.copy(w)
    dw[1:] = np.abs(np.diff(dw, n=1, axis=0))
    dw_count = int(dw.shape[0])

    hpp = np.ones((dw_count - window, filter_len))
    for i in range(window, dw.shape[0]):
        if i % 100 == 0:
            pass  # print((str(datetime.now())), " processing: ", i)
        for j in range(filter_len):
            poted_values = pot(dw[i - window:i, j], pot_method)
            if dw[i, j] > poted_values[-1]:
                fit = genpareto.fit(poted_values, floc=[poted_values[-1]])
                if dw[i, j] >= fit[1]:
                    hpp[i - window, j] = 1 - genpareto.cdf(dw[i, j], fit[0], fit[1], fit[2]) + 1e-20

    ese_value = -np.log10(np.prod(hpp, axis=1))
    return np.append(np.zeros(window), ese_value)
