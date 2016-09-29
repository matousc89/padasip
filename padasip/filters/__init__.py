"""
This sub-module stores adaptive filters and all related stuff.

An adaptive filter is a system that changes its adaptive parameteres
- adaptive weights :math:`\\textbf{w}(k)` - according to an optimization algorithm.

The parameters of all implemented adaptive filters can be:

* selected manually and passed to a filter as an array

* set to random - this will produce a vector of random values (zero mean,
    0.5 standard deviation)
    
* set to zeros

All filters can be called directly from padasip without further imports
as follows (LMS filter example):

    >>> import padasip as pa
    >>> pa.filters.FilterLMS(3, mu=1.)
    <padasip.filters.lms.FilterLMS instance at 0xb726edec>
    
"""
from padasip.filters.lms import FilterLMS
from padasip.filters.nlms import FilterNLMS
from padasip.filters.ocnlms import FilterOCNLMS
from padasip.filters.gngd import FilterGNGD
from padasip.filters.rls import FilterRLS
from padasip.filters.ap import FilterAP







