"""
.. versionadded:: 1.0.0

This module contatines tools for tasks of detection, known as:

* Novelty detection

* Fault Detection

* Outlier Detection

* System change point detection


Content of this page:

.. contents::
   :local:
   :depth: 1


Implemented tools
========================

.. toctree::
    :glob:
    :maxdepth: 1

    detection/*


"""
from padasip.detection.le import learning_entropy
from padasip.detection.elbnd import ELBND

