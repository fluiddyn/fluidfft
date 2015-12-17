Overview
========

Fast Fourier transforms (FFT) are useful for many applications.  Many good
libraries to perform FFT, in particular FFTW.  Recently, libraries to perform
FFT on clusters with the memory distributed over a large number of cores
(fftw3-mpi, pfft, p3dfft, 2decomp&FFT, ...).

There are already tools to perform FFT with Python (scipy.fftpack,
pyfftw). However, they suffer from drawbacks and many things are missing:

- nearly nothing for parallel FFT with distributed memory (using mpi),

- quite complicated even for the most simple and common cases. To understand
  anything, a novice user has to read at least the FFTW documentation.

- no benchmarks between libraries and between the Python solutions and
  solutions based only on a compiled language (as C, C++ or Fortran).

- just the FFT, no operator...
  
The Python library FluidFFT fills this gap by providing C++ classes and their
Python wrapper classes for performing simple and common tasks with different
libraries.  FluidFFT has been written to make things easy while being as
efficient as possible.  It provides:

- tests,

- documentation,

- benchmarks,

- operators for simple tasks (for example, compute the mean of the energy).
