.. _changelog:

Changelog
===========

**Version 1.2.2** *Released: 2022-08-05*
 Added new adaptive filters: :ref:`filter-vslms_mathews`,
 :ref:`filter-vslms_benveniste`, :ref:`filter-vslms_ang`.

**Version 1.2.1** *Released: 2022-02-07*
 Bugfix of the main adaptive filter class.

**Version 1.2.0** *Released: 2022-01-28*
 All adaptive filters were significantly refactored.
 Added new adaptive filters :ref:`filter-gmcc`, :ref:`filter-llncosh`
 and :ref:`filter-ocnlms`.
 Added new detection tool :ref:`detection-ese`.
 ANN module is removed from docs, and will be removed totally in future -
 there are much better Python libraries for ANN utilization.

**Version 1.1.1** *Released: 2017-08-06*
 Bugfix of adaptive filter helper function according to comments of
 `Patrick Bartels <https://github.com/pckbls>`_.

**Version 1.1.0** *Released: 2017-05-19*
 Added new adaptive filters :ref:`filter-lmf`, :ref:`filter-nlmf`,
 :ref:`filter-sslms` and :ref:`filter-nsslms`.

**Version 1.0.0** *Released: 2017-03-16*
 Added module :ref:`detection` containing :ref:`detection-le` and
 :ref:`detection-elbnd`.
 All implemented adaptive filters were updated. As a result,
 some obsolete helper functions for the adaptive filters were removed.
 Please use newer helper functions introduced in v0.7.
 Tutorials were updated and moved to `Python Adaptive Signal Processing Handbook
 <https://github.com/matousc89/Python-Adaptive-Signal-Processing-Handbook>`_.

**Version 0.7** *Released: 2017-01-07*
 Added new helper functions into  :ref:`filters`. Furthermore, the
 documentation for adaptive filters module was updated.
 Added functions for error evaluation - MSE, MAE, RMSE and logSE
 (:ref:`mics-error_evaluation`).

**Version 0.6** *Released: 2016-12-15*
 Added :ref:`preprocess-pca` and :ref:`preprocess-lda`. The whole documentation
 for preprocess module was updated.

**Version 0.5** *Released: 2016-11-16*
 Bugfix according to issue opened by https://github.com/lolpenguin

**Version 0.4** *Released: 2016-09-29*
 Added :ref:`filter-ap`. And also the first unit tests were implemented.

**Version 0.3** *Released: 2016-09-22*
 Added MLP into ANN module.

**Version 0.2** *Released: 2016-09-02*
 Added :ref:`filter-gngd`

**Version 0.1** *Released: 2016-03-18*
 Created
 :ref:`filter-lms`, :ref:`filter-nlms`,
 :ref:`filter-rls`, :ref:`preprocess-input_from_history`,
 :ref:`preprocess-standardize`, :ref:`preprocess-standardize_back`
