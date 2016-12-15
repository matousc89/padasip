"""
Current version: |version| (:ref:`changelog`)

This library is designed to simplify adaptive signal 
processing tasks within python
(filtering, prediction, reconstruction, classification).
For code optimisation,
this library uses `Numpy <http://www.numpy.org/>`_ for array operations.

Also in this library is presented some new methods
for adaptive signal processing.
The library is designed to be used with datasets and also with 
real-time measuring (sample-after-sample feeding).

.. toctree::
    :maxdepth: 2
    
    index

License
************

This project is under `MIT License <https://en.wikipedia.org/wiki/MIT_License>`_.

Instalation
************
With `pip <https://pypi.python.org/pypi/pip>`_ from terminal: ``$ pip install padasip``

Or download you can download the source codes from Github
(`link <https://github.com/matousc89/padasip>`_)


Tutorials
***********

All tutorials are created as Jupyter notebooks.
You can open the tutorial as html, or you can download it as notebook.

* Noise Cancelation, Identification and Prediction with Adaptive Filters
    (`see online <tutorials/tutorial1.html>`_) (`download notebook <notebooks/tutorial1.ipynb>`_)

* Adaptive Novelty Detection
    (`see online <tutorials/tutorial2.html>`_) (`download notebook <notebooks/tutorial2.ipynb>`_)
    
* Adaptive filters in Real-time with PADASIP Module
    (`see online <tutorials/tutorial3.html>`_) (`download notebook <notebooks/tutorial3.ipynb>`_)

* Multi-layer Perceptron (MLP) Neural Network - Basic Examples
    (`see online <tutorials/mlp_tutorial.html>`_) (`download notebook <notebooks/mlp_tutorial.ipynb>`_)


The User Quide
***************

If you need to know something what is not covered by tutorials,
check the complete documentation here



.. toctree::
    :maxdepth: 2
    
    sources/preprocess
    sources/filters_mod
    sources/ann


Contact
**********

By email: matousc@gmail.com


Indices and tables
*******************
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

"""
#from padasip.preprocess import 
from padasip.filters.shortcuts import *
import padasip.ann
import padasip.filters

import padasip.preprocess

# back compatibility with v0.5
from padasip.preprocess.standardize import standardize
from padasip.preprocess.standardize_back import standardize_back
from padasip.preprocess.input_from_history import input_from_history


