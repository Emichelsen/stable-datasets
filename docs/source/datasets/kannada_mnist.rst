:orphan:

Kannada-MNIST
=================

.. note::

   Dataset support is in progress. The loader is not available yet.

Overview
--------

Kannada-MNIST contains 70,000 28 x 28 grayscale images of handwritten
Kannada digits across 10 classes. The original split contains 60,000
training images and 10,000 test images.

The planned builder will use the author's MNIST-format files, preserve
the original train/test split, and expose ``image`` and ``label`` fields.
The separate Dig-MNIST dataset is outside the initial scope.
Loading examples and tests will be added with the implementation.

Source
------

- `Author's repository <https://github.com/vinayprabhu/Kannada_MNIST>`_
- `Dataset release <https://zenodo.org/records/3359691>`_
- License: Creative Commons Attribution 4.0 International (CC BY 4.0).
