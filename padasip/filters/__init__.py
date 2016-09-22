"""
This sub-module stores adaptive filters.

All filters can be called directly from padasip without further imports
as follows:

    >>> import padasip as pa
    >>> pa.filters.FilterLMS(3, mu=1.)
    <padasip.filters.lms.FilterLMS instance at 0xb726edec>

"""
from padasip.filters.lms import FilterLMS
from padasip.filters.nlms import FilterNLMS
from padasip.filters.ocnlms import FilterOCNLMS
from padasip.filters.gngd import FilterGNGD
from padasip.filters.rls import FilterRLS







