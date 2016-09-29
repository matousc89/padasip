"""
.. versionadded:: 0.1

All settings (constants) for the whole library are stored in this file.
These settings are often used in multiple files, thus why it is all in one file.
"""
### LMS adaptive filter

MU_LMS = 0.01
""" Default learning rate for LMS algorithm """

MU_LMS_MIN = 0.
""" Minimal allowed learning rate for LMS algorithm """

MU_LMS_MAX = 1000.
""" Maximal allowed learning rate for LMS algorithm """


### NLMS adaptive filter

MU_NLMS = 0.1
""" Default learning rate for NLMS algorithm """

MU_NLMS_MIN = 0.
""" Minimal allowed learning rate for NLMS algorithm """

MU_NLMS_MAX = 1000.
""" Minimal allowed learning rate for NLMS algorithm """

EPS_NLMS = 1.
""" Default regularization term for NLMS algorithm """

EPS_NLMS_MIN = 0.
""" Minimal allowed regularization term for NLMS algorithm """

EPS_NLMS_MAX = 1000.
""" Minimal allowed regularization term for NLMS algorithm """

### OCNLMS adaptive filter

MU_OCNLMS = 0.1
""" Default learning rate for OC-NLMS algorithm """

MU_OCNLMS_MIN = 0.
""" Minimal allowed learning rate for OC-NLMS algorithm """

MU_OCNLMS_MAX = 1000.
""" Minimal allowed learning rate for OC-NLMS algorithm """

EPS_OCNLMS = 1.
""" Default regularization term for OC-NLMS algorithm """

EPS_OCNLMS_MIN = 0.
""" Minimal allowed regularization term for OC-NLMS algorithm """

EPS_OCNLMS_MAX = 1000.
""" Minimal allowed regularization term for OC-NLMS algorithm """

MEM_OCNLMS = 100
""" Default memory size for OC-NLMS algorithm. """

##### RLS adaptive filter

MU_RLS = 0.99
""" Default learning rate for NLMS algorithm """

MU_RLS_MIN = 0.
""" Minimal allowed learning rate for NLMS algorithm """

MU_RLS_MAX = 1.
""" Minimal allowed learning rate for NLMS algorithm """

EPS_RLS = 0.1
""" Default regularization term for RLS algorithm """

EPS_RLS_MIN = 0.
""" Minimal allowed regularization term for RLS algorithm """

EPS_RLS_MAX = 1.
""" Minimal allowed regularization term for RLS algorithm """


##### GNGD adaptive filter

MU_GNGD = 1.
""" Default learning rate for NLMS algorithm """

MU_GNGD_MIN = 0.
""" Minimal allowed learning rate for NLMS algorithm """

MU_GNGD_MAX = 1000.
""" Minimal allowed learning rate for NLMS algorithm """


EPS_GNGD = 1.
""" Default regularization term for RLS algorithm """

EPS_GNGD_MIN = 0.
""" Minimal allowed regularization term for GNGD algorithm """

EPS_GNGD_MAX = 10.
""" Minimal allowed regularization term for GNGD algorithm """


RO_GNGD = 0.1
""" Default regularization term for GNGD algorithm """

RO_GNGD_MIN = 0.
""" Minimal allowed regularization term for GNGD algorithm """

RO_GNGD_MAX = 1.
""" Minimal allowed regularization term for GNGD algorithm """

### AP adaptive filter

MU_AP = 0.1
""" Default learning rate for AP algorithm """

MU_AP_MIN = 0.
""" Minimal allowed learning rate for AP algorithm """

MU_AP_MAX = 1000.
""" Minimal allowed learning rate for AP algorithm """

EPS_AP = 0.001
""" Default regularization term for AP algorithm """

EPS_AP_MIN = 0.
""" Minimal allowed regularization term for AP algorithm """

EPS_AP_MAX = 1000.
""" Minimal allowed regularization term for AP algorithm """


