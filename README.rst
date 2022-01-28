This library is designed to simplify adaptive signal
processing tasks within python
(filtering, prediction, reconstruction).
For code optimisation, this library uses numpy for array operations.

Also in this library is presented some new methods for adaptive signal processing.
The library is designed to be used with datasets and also with
real-time measuring (sample-after-sample feeding).

============================
Tutorials and Documentation
============================

Everything is on github:

http://matousc89.github.io/padasip/

================
Current Features
================

********************
Data Preprocessing
********************

- Principal Component Analysis (PCA)

- Linear Discriminant Analysis (LDA)

******************
Adaptive Filters
******************

The library features multiple adaptive filters. Input vectors for filters can be
constructed manually or with the assistance of included functions.
So far it is possible to use following filters:

- LMS (least-mean-squares) adaptive filter

- NLMS (normalized least-mean-squares) adaptive filter

- LMF (least-mean-fourth) adaptive filter

- NLMF (normalized least-mean-fourth) adaptive filter

- SSLMS (sign-sign least-mean-squares) adaptive filter

- NSSLMS (normalized sign-sign least-mean-squares) adaptive filter

- RLS (recursive-least-squares) adaptive filter

- GNGD (generalized normalized gradient descent) adaptive filter

- AP (affine projection) adaptive filter

- GMCC (generalized maximum correntropy criterion) adaptive filter

- OCNLMS (online centered normalized least-mean-squares) adaptive filter

- Llncosh (least lncosh) adaptive filter

******************
Detection Tools
******************

The library features two novelty/outlier detection tools

- Error and Learning Based Novelty Detection (ELBND)

- Learning Entropy (LE)

- Extreme Seeking Entropy (ESE)
