FluidFFT: Efficient and easy Fast Fourier Transform for Python
==============================================================

|release| |docs| |coverage|

.. |release| image:: https://img.shields.io/pypi/v/fluidfft.svg
   :target: https://pypi.python.org/pypi/fluidfft/
   :alt: Latest version

.. |docs| image:: https://readthedocs.org/projects/fluidfft/badge/?version=latest
   :target: http://fluidfft.readthedocs.org
   :alt: Documentation status

.. |coverage| image:: https://codecov.io/bb/fluiddyn/fluidfft/branch/default/graph/badge.svg
   :target: https://codecov.io/bb/fluiddyn/fluidfft/branch/default/
   :alt: Code coverage

This package provides C++ and Python classes useful to perform fast
Fourier transform with different libraries, in particular

- fftw3 and fftw3-mpi
  
- `pfft <https://github.com/mpip/pfft>`_

- `p3dfft <https://github.com/sdsc/p3dfft>`_
  
The package can be used as a base to write spectral code. In
particular, the code `FluidSim <http://fluidsim.readthedocs.org>`_
**will** use it.
