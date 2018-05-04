.. FluidFFT documentation master file

FluidFFT documentation
======================

This package provides C++ classes and their Python wrapper classes useful to
perform Fast Fourier Transform (FFT) with different libraries, in particular

- `fftw3 <http://www.fftw.org/>`_ and `fftw3-mpi
  <http://www.fftw.org/fftw3_doc/Distributed_002dmemory-FFTW-with-MPI.html>`_

- `pfft <https://github.com/mpip/pfft>`_

- `p3dfft <https://github.com/sdsc/p3dfft>`_

- `cufft <https://developer.nvidia.com/cufft>`_ (fft library by CUDA
  running on GPU)

`pfft <https://github.com/mpip/pfft>`_ and `p3dfft
<https://github.com/sdsc/p3dfft>`_ are specialized in computing FFT efficiently
on several cores of big clusters. The data is split in pencils and the
computations can be distributed on several processes.

FluidFFT provides an unified API to use all these libraries. FluidFFT is not
limited to just performing Fourier transforms. It is a complete development
framework for codes using (distributed) FFT. A simple API allows the developers to
easily perform operations on data in real and spectral spaces (gradient,
divergence, rotational, sum over wavenumbers, computation of spectra, etc.) and
deal with the data distribution (gather the data on one process and scatter the
data to many processes) without having to consider the internal organization of
every FFT library.

FluidFFT has been created to be:

- Easy to install (see :ref:`install`).

- Easy to use, both for simple users and for developers (see
  :ref:`tuto`). FluidFFT hides the internal complication of (distributed) FFT
  libraries.

- Robust (unittest coverage larger than 90 %).

- Highly efficient.

  The architecture of the code and the tools used (C++ classes, Cython wrapper and
  `Pythran <https://github.com/serge-sans-paille/pythran>`_ computational
  functions) leads to very good performances.

  Moreover, Python developers can easily measure the performance cost of using
  Python compared to coding in pure C++.  This cost has to be compared to the cost
  of the FFT in pure C++, which of course strongly varies with the size of the
  arrays.  From our experience with real cases, the "Python cost" goes from very
  small (for small cases) to completely negligible (for medium and large cases).

  A great advantage of FluidFFT is that it allows the user to find (see
  :ref:`bench`) and to choose the most efficient solution for a particular
  case.  Since the fastest library depends on the case and on the hardware, it
  is really a useful feature for performance!

FluidFFT is therefore a very useful tool to write HPC applications using FFT, as
for example pseudo-spectral simulation codes.  For an example of how FluidFFT can
be used in a real application, see `the code
<https://bitbucket.org/fluiddyn/fluidsim>`_ of the `Computational Fluid Dynamics
(CFD) framework FluidSim <http://fluidsim.readthedocs.org>`_.


User Guide
----------

.. toctree::
   :maxdepth: 2

   overview
   install
   tutorials
   examples
   bench


Modules Reference
-----------------

.. autosummary::
   :toctree: generated/

   fluidfft

See also the `documentation of the cpp code produced with Doxygen
<doxygen/index.html>`_


More
----

.. toctree::
   :maxdepth: 1

   changes
   Advice for FluidDyn developers <http://fluiddyn.readthedocs.io/en/latest/advice_developers.html>
   to_do


Links
-----

.. |release| image:: https://img.shields.io/pypi/v/fluidfft.svg
   :target: https://pypi.org/project/fluidfft/
   :alt: Latest version

.. |coverage| image:: https://codecov.io/bb/fluiddyn/fluidfft/branch/default/graph/badge.svg
   :target: https://codecov.io/bb/fluiddyn/fluidfft
   :alt: Code coverage

.. |travis| image:: https://travis-ci.org/fluiddyn/fluidfft.svg?branch=master
   :target: https://travis-ci.org/fluiddyn/fluidfft

.. |pipelines| image:: https://img.shields.io/bitbucket/pipelines/fluiddyn/fluidfft.svg
   :target: https://bitbucket.org/fluiddyn/fluidfft/addon/pipelines/home#!/

- `FluidDyn documentation <http://fluiddyn.readthedocs.io>`_
- `FluidFFT forge on Bitbucket <https://bitbucket.org/fluiddyn/fluidfft>`_
- FluidFFT in PyPI |release|
- Unittest coverage |coverage|
- Continuous integration with travis-ci.org |travis|
- Continuous integration with Bitbucket Pipelines |pipelines|
- `FluidDyn user chat room
  <https://riot.im/app/#/room/#fluiddyn-users:matrix.org>`_
- `FluidDyn mailing list <https://www.freelists.org/list/fluiddyn>`_
- `FluidDyn on Twitter <https://twitter.com/pyfluiddyn>`_


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
