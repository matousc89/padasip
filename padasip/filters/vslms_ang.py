"""
.. versionadded:: 1.2.2

The variable step-size least-mean-square (VSLMS) adaptive filter with Ang's adaptation
is implemeted according to
`DOI:10.1109/78.912925 <https://doi.org/10.1109/78.912925>`_.


The VSLMS filter with Benveniste adaptation can be created as follows

    >>> import padasip as pa
    >>> pa.filters.FilterVSLMS_Ang(n)

where `n` is the size (number of taps) of the filter.

Content of this page:

.. contents::
   :local:
   :depth: 1

.. seealso:: :ref:`filters`


Minimal Working Examples
======================================

If you have measured data you may filter it as follows

.. code-block:: python

    import numpy as np
    import matplotlib.pylab as plt
    import padasip as pa

    # creation of data
    N = 500
    x = np.random.normal(0, 1, (N, 4))  # input matrix
    v = np.random.normal(0, 0.1, N)  # noise
    d = 2 * x[:, 0] + 0.1 * x[:, 1] - 4 * x[:, 2] + 0.5 * x[:, 3] + v  # target

    # identification
    f = pa.filters.FilterVSLMS_Ang(n=4, mu=0.1, ro=0.0002, w="random")
    y, e, w = f.run(d, x)

    # show results
    plt.figure(figsize=(15, 9))
    plt.subplot(211);
    plt.title("Adaptation");
    plt.xlabel("samples - k")
    plt.plot(d, "b", label="d - target")
    plt.plot(y, "g", label="y - output");
    plt.legend()
    plt.subplot(212);
    plt.title("Filter error");
    plt.xlabel("samples - k")
    plt.plot(10 * np.log10(e ** 2), "r", label="e - error [dB]");
    plt.legend()
    plt.tight_layout()
    plt.show()


Code Explanation
======================================
"""
import numpy as np

from padasip.filters.base_filter import AdaptiveFilter


class FilterVSLMS_Ang(AdaptiveFilter):
    """
    This class represents an adaptive VSLMS filter with Ang's adaptation.
    """
    kind = "VSLMS_Ang"

    def __init__(self, n, mu=1., ro=0.0002, a=0.95, **kwargs):
        """
        **Kwargs:**

        * `ro` : step size adaptation parameter (float) at the beginning.
          It is an adaptive parameter.

        * `a` : small constant close to 1 (but smaller). It works as a simplification
          of the Benveniste's algoritm.

        """
        super().__init__(n, mu, **kwargs)
        self.ro = ro
        self.a = a
        self.last_e = 0
        self.last_x = np.zeros(n)
        self.last_fi = np.zeros(n)
        self.last_mu = mu

    def learning_rule(self, e, x):
        """
        Override the parent class.
        """
        fi = (self.a * self.last_fi) + (self.last_e * self.last_x)
        mu = self.last_mu + (self.ro * e * np.dot(self.last_x, fi))
        self.last_e, self.last_mu, self.last_x, self.last_fi = e, mu, x, fi
        return mu * e * x
