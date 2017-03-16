"""
.. versionadded:: 0.6

Linear discriminant analysis (LDA) :cite:`fisher1936use`
is a method used to determine the features
that separates some classes of items. The output of LDA may be used as
a linear classifier, or for dimensionality reduction for purposes of
classification.

.. contents::
   :local:
   :depth: 1
   
See also: :ref:`preprocess-pca`

Usage Explanation
********************

For reduction of data-set :code:`x` with labels stored in array (:code:`labels`)
to new dataset :code:`new_x` containg just :code:`n` number of
columns

.. code-block:: python

    new_x = pa.preprocess.LDA(x, labels, n) 
    
The sorted array of scattermatrix eigenvalues for dataset :code:`x` described
with variable :code:`labels` can be obtained as follows
    
.. code-block:: python

    eigenvalues = pa.preprocess.LDA_discriminants(x, labels) 
    

Minimal Working Examples
*****************************

In this example we create data-set :code:`x` of 150 random samples. Every sample
is described by 4 values and label. The labels are stored in 
array :code:`labels`.

Firstly, it is good to see the eigenvalues of scatter matrix to determine
how many rows is reasonable to reduce 

.. code-block:: python

    import numpy as np
    import padasip as pa

    np.random.seed(100) # constant seed to keep the results consistent

    N = 150 # number of samples
    classes = np.array(["1", "a", 3]) # names of classes
    cols = 4 # number of features (columns in dataset)

    x = np.random.random((N, cols)) # random data
    labels = np.random.choice(classes, size=N) # random labels

    print pa.preprocess.LDA_discriminants(x, labels)

what prints

>>> [  2.90863957e-02   2.28352079e-02   1.23545720e-18  -1.61163011e-18]

From this output it is obvious that reasonable number of columns to keep is 2.
The following code reduce the number of features to 2.

.. code-block:: python

    import numpy as np
    import padasip as pa
    
    np.random.seed(100) # constant seed to keep the results consistent

    N = 150 # number of samples
    classes = np.array(["1", "a", 3]) # names of classes
    cols = 4 # number of features (columns in dataset)

    x = np.random.random((N, cols)) # random data
    labels = np.random.choice(classes, size=N) # random labels

    new_x = pa.preprocess.LDA(x, labels, n=2)

to check if the size of new data-set is really correct we can print the shapes
as follows      

>>> print "Shape of original dataset: {}".format(x.shape) 
Shape of original dataset: (150, 4)
>>> print "Shape of new dataset: {}".format(new_x.shape)
Shape of new dataset: (150, 2)

References
***************

.. bibliography:: lda.bib
    :style: plain

Code Explanation
***************** 
"""
from __future__ import division
import numpy as np

def LDA_base(x, labels):
    """
    Base function used for Linear Discriminant Analysis.

    **Args:**

    * `x` : input matrix (2d array), every row represents new sample

    * `labels` : list of labels (iterable), every item should be label for \
      sample with corresponding index

    **Returns:**
    
    * `eigenvalues`, `eigenvectors` : eigenvalues and eigenvectors \
      from LDA analysis 

    """
    classes = np.array(tuple(set(labels)))
    cols = x.shape[1]
    # mean values for every class
    means = np.zeros((len(classes), cols))
    for i, cl in enumerate(classes):
        means[i] = np.mean(x[labels==cl], axis=0)
    # scatter matrices
    scatter_within = np.zeros((cols, cols))
    for cl, mean in zip(classes, means):
        scatter_class = np.zeros((cols, cols))
        for row in x[labels == cl]:
            dif = row - mean
            scatter_class += np.dot(dif.reshape(cols, 1), dif.reshape(1, cols))
        scatter_within += scatter_class
    total_mean = np.mean(x, axis=0)
    scatter_between = np.zeros((cols, cols))
    for cl, mean in zip(classes, means):
        dif = mean - total_mean
        dif_product = np.dot(dif.reshape(cols, 1), dif.reshape(1, cols))
        scatter_between += x[labels == cl, :].shape[0] * dif_product
    # eigenvalues and eigenvectors from scatter matrices
    scatter_product = np.dot(np.linalg.inv(scatter_within), scatter_between)
    eigen_values, eigen_vectors = np.linalg.eig(scatter_product)
    return eigen_values, eigen_vectors

def LDA(x, labels, n=False):
    """
    Linear Discriminant Analysis function.

    **Args:**

    * `x` : input matrix (2d array), every row represents new sample

    * `labels` : list of labels (iterable), every item should be label for \
      sample with corresponding index

    **Kwargs:**

    * `n` : number of features returned (integer) - how many columns 
      should the output keep

    **Returns:**
    
    * new_x : matrix with reduced size (number of columns are equal `n`)
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
    # make the LDA
    eigen_values, eigen_vectors = LDA_base(x, labels)
    # sort the eigen vectors according to eigen values
    eigen_order = eigen_vectors.T[(-eigen_values).argsort()]
    return eigen_order[:n].dot(x.T).T


def LDA_discriminants(x, labels):
    """
    Linear Discriminant Analysis helper for determination how many columns of
    data should be reduced.

    **Args:**

    * `x` : input matrix (2d array), every row represents new sample

    * `labels` : list of labels (iterable), every item should be label for \
        sample with corresponding index

    **Returns:**
    
    * `discriminants` : array of eigenvalues sorted in descending order

    """
    # validate inputs
    try:    
        x = np.array(x)
    except:
        raise ValueError('Impossible to convert x to a numpy array.')
    # make the LDA
    eigen_values, eigen_vectors = LDA_base(x, labels)
    return eigen_values[(-eigen_values).argsort()]

