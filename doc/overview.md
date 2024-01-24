# Overview

Fast Fourier transforms (FFT) are useful for many applications.  There are many
good libraries to perform FFT, in particular the standard FFTW.  A new
challenge is to perform efficiently FFT on clusters with the memory distributed
over a large number of cores. A problem is that for one-dimensional FFT, all
the data has to be located in the memory of the process that perform the FFT,
so a lot of communication between processes are needed for 2D and 3D FFT.

There are two strategies to distribute the memory, the 1D (or "slab")
decomposition and the 2D (or "pencil") decomposition. The 1D decomposition is
more efficient when only few processes are used but suffers from an important
limitation in terms of number of MPI processes that can be used. In contrast,
this limitation is overcome by the 2D decomposition.  FFTW supports MPI using
1D decomposition and hybrid parallelism using OpenMP. Other libraries now
implement the 2D decomposition: pfft, p3dfft, 2decomp&FFT, ... These libraries
rely on MPI for the communications between processes, are optimized for
supercomputers and scales well to hundreds of thousands of cores. However,
since there is no common API, it is not simple to write applications that are
able to use these libraries and to compare their performances.

There are already tools to perform FFT with Python (scipy.fftpack,
pyfftw). However, they suffer from drawbacks:

- nearly nothing for parallel FFT with distributed memory (using mpi),
- quite complicated even for the most simple and common cases. To understand how
  to use them, a novice user has to read at least the [FFTW documentation](http://www.fftw.org/fftw3_doc/).
- no benchmarks between libraries and between the Python solutions and solutions
  based only on a static language (as C, C++ or Fortran).
- just the FFT, no linear operators and utilities to deal with the data
  distribution...

The Python package FluidFFT fills this gap by providing C++ classes and their
Python wrapper classes for performing simple and common tasks with different
FFT libraries.  FluidFFT has been written to make things easy while being as
efficient as possible.  It provides:

- tests,
- documentation,
- benchmarks,
- operators for simple tasks (for example, compute the energy or the gradient
  of a field).
