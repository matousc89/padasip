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
===============

This project is under `MIT License <https://en.wikipedia.org/wiki/MIT_License>`_.

Instalation
====================
With `pip <https://pypi.python.org/pypi/pip>`_ from terminal: ``$ pip install padasip``

Or download you can download the source codes from Github
(`link <https://github.com/matousc89/padasip>`_)


Tutorials
===============

All Padasip related tutorials are created as Jupyter notebooks. You can find
them in `Python Adaptive Signal Processing Handbook
<https://github.com/matousc89/Python-Adaptive-Signal-Processing-Handbook>`_.

The User Quide
=====================

If you need to know something what is not covered by tutorials,
check the complete documentation here


.. toctree::
    :maxdepth: 2
    :titlesonly:
    
    sources/preprocess
    sources/filters
    sources/ann
    sources/detection
    sources/misc


Contact
=====================

By email: matousc@gmail.com


Changelog
======================

For informations about versions and updates see :ref:`changelog`.

Indices and tables
===========================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

"""
#from padasip.preprocess import 
import padasip.ann
import padasip.filters
import padasip.preprocess
import padasip.misc
import padasip.detection

# back compatibility with v0.5
from padasip.preprocess.standardize import standardize
from padasip.preprocess.standardize_back import standardize_back
from padasip.preprocess.input_from_history import input_from_history


