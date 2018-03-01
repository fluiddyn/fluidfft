FluidFFT: efficient and easy Fast Fourier Transform for Python
==============================================================

|release| |docs| |coverage| |travis|

.. |release| image:: https://img.shields.io/pypi/v/fluidfft.svg
   :target: https://pypi.org/project/fluidfft/
   :alt: Latest version

.. |docs| image:: https://readthedocs.org/projects/fluidfft/badge/?version=latest
   :target: http://fluidfft.readthedocs.org
   :alt: Documentation status

.. |coverage| image:: https://codecov.io/bb/fluiddyn/fluidfft/branch/default/graph/badge.svg
   :target: https://codecov.io/bb/fluiddyn/fluidfft
   :alt: Code coverage

.. |travis| image:: https://travis-ci.org/fluiddyn/fluidfft.svg?branch=master
    :target: https://travis-ci.org/fluiddyn/fluidfft

This package provides C++ classes and their Python wrapper classes written in
Cython useful to perform Fast Fourier Transform (FFT) with different libraries,
in particular

- `fftw3 <http://www.fftw.org/>`_ and `fftw3-mpi
  <http://www.fftw.org/fftw3_doc/Distributed_002dmemory-FFTW-with-MPI.html>`_
  
- `pfft <https://github.com/mpip/pfft>`_

- `p3dfft <https://github.com/sdsc/p3dfft>`_
    
- `cufft <https://developer.nvidia.com/cufft>`_ (fft library by CUDA
  running on GPU)

`pfft <https://github.com/mpip/pfft>`_ and `p3dfft
<https://github.com/sdsc/p3dfft>`_ are specialized in computing FFT efficiently
on several cores of big clusters. The data is split in pencils and can be
distributed on several processes.

Fluidfft provides classes to use in a transparent way all these libraries with
an unified API. These classes are not limited to just performing Fourier
transforms. They are also an elegant solution to efficiently perform operations
on data in real and spectral spaces (gradient, divergence, rotational, sum over
wavenumbers, computation of spectra, etc.) and easily deal with the data
distribution (gather the data on one process, scatter the data to many
processes) without having to know the internal organization of every FFT
library.

Fluidfft hides the internal complication of (distributed) FFT libraries and
allows the user to find (by benchmarking) and to choose the most efficient
solution for a particular case. Fluidfft is therefore a very useful tool to
write HPC applications using FFT, as for example pseudo-spectral simulation
codes. In particular, fluidfft is used in the Computational Fluid Dynamics
(CFD) framework `fluidsim <http://fluidsim.readthedocs.org>`_.
