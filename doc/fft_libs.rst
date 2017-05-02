Supported FFT libraries and their installation
==============================================

.. todo::

   Add more on building of FFT libraries.

fftw (better to build with OpenMP enabled)
------------------------------------------

`FFTW <http://www.fftw.org/>`_ (*Fastest Fourier Transform in the West*) is the
standard open-source library for discrete Fourier transform.

.. todo::

   Use OpenMP...
    
fftw-mpi
--------

`FFTW <http://www.fftw.org/>`_ provide their parallel MPI implementation.

MKL library (FFT by intel)
--------------------------

.. todo:: 

   Try to use MKL wrappers. Note that there are wrappers to use the MKL library
   (FFT by intel) using the FFTW API.

`pfft <https://github.com/mpip/pfft>`_ 
--------------------------------------

*"PFFT is a software library for computing massively parallel, fast Fourier
transformations on distributed memory architectures. PFFT can be understood as
a generalization of FFTW-MPI to multidimensional data decomposition. The
library is written in C and MPI."*

`p3dfft <https://www.p3dfft.net/>`_
-----------------------------------

*"Parallel Three-Dimensional Fast Fourier Transforms, dubbed P3DFFT, is a
library for large-scale computer simulations on parallel platforms. [...]
P3DFFT uses 2D, or pencil, decomposition. This overcomes an important
limitation to scalability inherent in FFT libraries implementing 1D (or slab)
decomposition: the number of processors/tasks used to run this problem in
parallel can be as large as N2, where N is the linear problem size."*

cuda
----
  
Modify fftw.h ! https://github.com/FFTW/fftw3/issues/18


