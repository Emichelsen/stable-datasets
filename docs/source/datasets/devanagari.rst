:orphan:

Devanagari Handwritten Character Dataset
============================================

.. note::

   Dataset support is in progress. The loader is not available yet.

Overview
--------

The Devanagari Handwritten Character Dataset (DHCD) contains 92,000
32 x 32 grayscale images across 46 classes: 36 characters and 10 digits.
The original split contains 78,200 training images and 13,800 test images.

The planned builder will preserve the original train/test split and expose
``image`` and ``label`` fields. Documentation of the class mapping, loading
examples, and tests will be added with the implementation.

Source
------

- `UCI dataset page <https://archive.ics.uci.edu/dataset/389/devanagari+handwritten+character+dataset>`_
- License: Creative Commons Attribution 4.0 International (CC BY 4.0).
