FluidFFT: Efficient and easy Fast Fourier Transform for Python
==============================================================

This package provides C++ and Python classes useful to perform fast
Fourier transform with different libraries, in particular

- fftw3 and fftw3-mpi
  
- `pfft <https://github.com/mpip/pfft>`_

- `p3dfft <https://github.com/sdsc/p3dfft>`_ (in dev...)
  
The package can be used as a base to write spectral code. In
particular, the code `FluidSim <http://fluidsim.readthedocs.org>`_
**will** use it.
