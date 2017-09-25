FluidFFT: efficient and easy Fast Fourier Transform for Python
==============================================================

|release| |docs| |coverage| |travis|

.. |release| image:: https://img.shields.io/pypi/v/fluidfft.svg
   :target: https://pypi.python.org/pypi/fluidfft/
   :alt: Latest version

.. |docs| image:: https://readthedocs.org/projects/fluidfft/badge/?version=latest
   :target: http://fluidfft.readthedocs.org
   :alt: Documentation status

.. |coverage| image:: https://codecov.io/gh/fluiddyn/fluidfft/graph/badge.svg
   :target: https://codecov.io/gh/fluiddyn/fluidfft
   :alt: Code coverage

.. |travis| image:: https://travis-ci.org/fluiddyn/fluidfft.svg?branch=master
    :target: https://travis-ci.org/fluiddyn/fluidfft

This package provides C++ classes and their Python wrapper classes written in
Cython useful to perform Fast Fourier Transform (FFT) with different libraries,
in particular

- fftw3 and fftw3-mpi

- `pfft <https://github.com/mpip/pfft>`_

- `p3dfft <https://github.com/sdsc/p3dfft>`_

- `cufft <https://developer.nvidia.com/cufft>`_ (fft library by CUDA
  running on GPU)

`pfft <https://github.com/mpip/pfft>`_ and `p3dfft
<https://github.com/sdsc/p3dfft>`_ are specialized in computing FFT efficiently
on several cores of big clusters. The data is spit in pencils and can be
distributed on several processes.

Fluidfft provides an uniform API for all these libraries. Fluidfft can be used
for example as a base to write pseudo-spectral simulation codes. In particular,
fluidfft is used in the code `fluidsim
<http://fluidsim.readthedocs.org>`_. Fluidfft also provides classes to
efficiently perform linear operators (gradient, divergence, rotational, etc.)
and easily deal with the data distribution (gather the data on one process,
scatter the data to many processes).
