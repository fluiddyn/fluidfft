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

If a system-wide installation of FFTW with OpenMP and MPI is not available,
the user can make a local installation in a user-accessible directory. You can
utilize this bash script with some modifications to the first few lines to
install the FFTW library.

.. literalinclude:: install_fftw.sh
   :language: shell


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

.. todo:: 

   How can I install it? Link? Advice?

`p3dfft <https://www.p3dfft.net/>`_
-----------------------------------

*"Parallel Three-Dimensional Fast Fourier Transforms, dubbed P3DFFT, is a
library for large-scale computer simulations on parallel platforms. [...]
P3DFFT uses 2D, or pencil, decomposition. This overcomes an important
limitation to scalability inherent in FFT libraries implementing 1D (or slab)
decomposition: the number of processors/tasks used to run this problem in
parallel can be as large as N2, where N is the linear problem size."*

To use p3dfft with python binding, we need shared library and then a modified
version of the package. Please follow the process detailed in the INSTALL file
of this fork of the official p3dfft package : 
https://github.com/CyrilleBonamy/p3dfft

It can be convenient to link p3dfft with fftw3. So one fftw directory with lib 
and include directories must exist.

For example, on debian system, you can do that with :
mkdir -p $HOME/opt/fft_gcc/include/
mkdir -p $HOME/opt/fft_gcc/lib/
cp /usr/include/fftw* $HOME/opt/fft_gcc/include/.
cp /usr/lib/x86_64-linux-gnu/libfftw3* $HOME/opt/fft_gcc/lib/.

And next you can compile p3dfft with the following command
(in this case the final install directory is /opt/p3dfft/2.7.5 directory) :
CC=mpicc CCLD=mpif90 ./configure --enable-fftw --with-fftw=$HOME/opt/fft_gcc_bak/ --prefix=/opt/p3dfft/2.7.5

.. todo:: 

   How can I install it? Link? Advice?

cuda
----

Modify fftw.h ! https://github.com/FFTW/fftw3/issues/18


.. todo:: 

   How can I install it? Link? Advice?
