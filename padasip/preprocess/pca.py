"""
.. versionadded:: 0.6

Principal component analysis (PCA) is a statistical method how to convert a set of observations with possibly correlated variables into a data-set of linearly uncorrelated variables (principal components). The number of principal components is less or equal than the number of original variables. This transformation is defined in such a way that the first principal component has the largest possible variance.

.. contents::
   :local:
   :depth: 1
   
See also: :ref:`preprocess-lda`

Usage Explanation
=======================

For reduction of dataset :code:`x` to :code:`n` number of principal components

.. code-block:: python

    new_x = pa.preprocess.PCA(x, n) 

If you want to see the ordered eigenvalues of principal components, 
you can do it as follows:

.. code-block:: python

    eigenvalues = pa.preprocess.PCA_components(x) 

Minimal Working Example
===========================

In this example is generated random numbers (100 samples, with 3 values each).
After the PCA application the reduced data-set is produced
(all samples, but only 2 valueseach)

.. code-block:: python

    import numpy as np
    import padasip as pa

    np.random.seed(100) 
    x = np.random.uniform(1, 10, (100, 3))
    new_x = pa.preprocess.PCA(x, 2) 

If you do not know, how many principal components you should use,
you can check the eigenvalues of principal components according to 
following example 

.. code-block:: python

    import numpy as np
    import padasip as pa

    np.random.seed(100) 
    x = np.random.uniform(1, 10, (100, 3))
    print pa.preprocess.PCA_components(x) 

what prints 

>>> [ 8.02948402  7.09335781  5.34116273]

Code Explanation
====================
"""
from __future__ import division
import numpy as np


def PCA_components(x):
    """
    Principal Component Analysis helper to check out eigenvalues of components.

    **Args:**

    * `x` : input matrix (2d array), every row represents new sample

    **Returns:**
    
    * `components`: sorted array of principal components eigenvalues 
        
    """ 
    # validate inputs
    try:    
        x = np.array(x)
    except:
        raise ValueError('Impossible to convert x to a numpy array.')
    # eigen values and eigen vectors of data covariance matrix
    eigen_values, eigen_vectors = np.linalg.eig(np.cov(x.T))
    # sort eigen vectors according biggest eigen value
    eigen_order = eigen_vectors.T[(-eigen_values).argsort()]
    # form output - order the eigenvalues
    return eigen_values[(-eigen_values).argsort()]


def PCA(x, n=False):
    """
    Principal component analysis function.

    **Args:**

    * `x` : input matrix (2d array), every row represents new sample

    **Kwargs:**

    * `n` : number of features returned (integer) - how many columns 
      should the output keep

    **Returns:**
    
    * `new_x` : matrix with reduced size (lower number of columns)
    """
    # select n if not provided
    if not n:
        n = x.shape[1] - 1   
    # validate inputs
    try:    
        x = np.array(x)
    except:
        raise ValueError('Impossible to convert x to a numpy array.')
    assert type(n) == int, "Provided n is not an integer."
    assert x.shape[1] > n, "The requested n is bigger than \
        number of features in x."
    # eigen values and eigen vectors of data covariance matrix
    eigen_values, eigen_vectors = np.linalg.eig(np.cov(x.T))
    # sort eigen vectors according biggest eigen value
    eigen_order = eigen_vectors.T[(-eigen_values).argsort()]
    # form output - reduced x matrix
    return eigen_order[:n].dot(x.T).T




