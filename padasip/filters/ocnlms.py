"""
.. versionadded:: 1.2.0

The online-centered normalized least-mean-squares (OCNLMS) adaptive filter
(proposed in https://doi.org/10.14311/nnw.2021.31.019)
is an extension of the popular NLMS adaptive filter (:ref:`filter-nlms`).

The OCNLMS filter can be created as follows

    >>> import padasip as pa
    >>> pa.filters.FilterOCNLMS(n, mem=100)

where `n` is the size (number of taps) of the filter.

Content of this page:

.. contents::
   :local:
   :depth: 1

.. seealso:: :ref:`filters`

Algorithm Explanation
======================================

The OCNLMS is extension of NLMS filter. See :ref:`filter-nlms`
for explanation of the algorithm behind.  As an extension of the
normalized least mean squares (NLMS), the OCNLMS algorithm
features an approach of online input centering according to
the introduced filter memory. This key feature can compensate
the effect of concept drift in data streams, because
such a centering makes the filter independent
of the nonzero mean value of signal.

Minimal Working Examples
======================================

Exampleof an unknown system identification from mesaured data follows.
The memory size `mem` is defined during the construction of the filter.

.. code-block:: python

    import numpy as np
    import matplotlib.pylab as plt
    import padasip as pa

    # creation of data
    N = 500
    x = np.random.normal(0, 1, (N, 4)) + 121 # input matrix with offset
    v = np.random.normal(0, 0.1, N) # noise
    d = 2*x[:,0] + 0.1*x[:,1] - 4*x[:,2] + 0.5*x[:,3] + v # target

    # identification, memory is set to 100 samples
    f = pa.filters.FilterOCNLMS(n=4, mu=0.1, w="random", mem=100)
    y, e, w = f.run(d, x)

    # show results
    plt.figure(figsize=(15,9))
    plt.subplot(211);plt.title("Adaptation");plt.xlabel("samples - k")
    plt.plot(d,"b", label="d - target")
    plt.plot(y,"g", label="y - output");plt.legend()
    plt.subplot(212);plt.title("Filter error");plt.xlabel("samples - k")
    plt.plot(10*np.log10(e**2),"r", label="e - error [dB]");plt.legend()
    plt.tight_layout()
    plt.show()


Code Explanation
======================================
"""
import numpy as np

from padasip.filters.base_filter import AdaptiveFilter

class FilterOCNLMS(AdaptiveFilter):
    """
    Adaptive OCNLMS filter.
    """
    kind = "OCNLMS"

    def __init__(self, n, mu=0.1, eps=1., mem=100, **kwargs):
        """
        Kwargs:

        * `eps` : regularization term (float). It is introduced to preserve
            stability for close-to-zero input vectors

        * `mem` : size of filter memory (int). This means how many last targets
            and input vectors will be used for centering of current input vector
            and target.
        """
        super().__init__(n, mu, **kwargs)
        self.eps = eps
        self.mem = mem
        self.clear_memory()

    def learning_rule(self, e, x):
        """
        Override the parent class.
        """
        self.update_memory_x(x)
        m_d, m_x = self.read_memory()
        y = np.dot(self.w, x-m_x) + m_d
        # e = d - y
        self.update_memory_d(e + y)
        return self.mu / (self.eps + np.dot(x - m_x, x - m_x)) * e * (x - m_x)

    def predict(self, x):
        """
        This function calculates OCNLMS specific output.
        The parent class `predict` function cannot be used.

        **Args:**

        * `x` : input vector (1 dimension array) in length of filter.

        **Returns:**

        * `y` : output value (float) calculated from input array.

        """
        m_d, m_x = self.read_memory()
        return np.dot(self.w, x - m_x) + m_d

    def clear_memory(self):
        """
        Clear of data from memory and reset memory index.
        """
        self.mem_empty = True
        self.mem_x = np.zeros((self.mem, self.n))
        self.mem_d = np.zeros(self.mem)
        self.mem_idx = 0

    def update_memory_d(self, d_k):
        """
        This function update memory of the filter with new target value `d`.
        """
        self.mem_d[self.mem_idx-1] = d_k

    def update_memory_x(self, x_k):
        """
        This function update memory of the filter with new input vector `x`.
        """
        self.mem_x[self.mem_idx, :] = x_k

    def read_memory(self):
        """
        This function read mean value of target`d`
        and input vector `x` from history
        """
        if self.mem_empty == True:
            if self.mem_idx == 0:
                m_x = np.zeros(self.n)
                m_d = 0
            else:
                m_x = np.mean(self.mem_x[:self.mem_idx+1], axis=0)
                m_d = np.mean(self.mem_d[:self.mem_idx])
        else:
            m_x = np.mean(self.mem_x, axis=0)
            m_d = np.mean(np.delete(self.mem_d, self.mem_idx))
        self.mem_idx += 1
        if self.mem_idx > len(self.mem_x)-1:
            self.mem_idx = 0
            self.mem_empty = False
        return m_d, m_x
