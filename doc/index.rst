.. FluidFFT documentation master file

FluidFFT documentation
======================

This package provides C++ classes and their Python wrapper classes useful to
perform Fast Fourier Transform (FFT) with different libraries, in particular

- fftw3 and fftw3-mpi
  
- `pfft <https://github.com/mpip/pfft>`_

- `p3dfft <https://github.com/sdsc/p3dfft>`_ (in dev...)
  
The package can be used for example as a base to write spectral simulation
code. In particular, the code `FluidSim <http://fluidsim.readthedocs.org>`_
**will** use it.


User Guide
----------

.. toctree::
   :maxdepth: 2

   overview
   install


Modules Reference
-----------------

.. autosummary::
   :toctree: generated/

   fluidfft2d
   fluidfft3d


More
----

.. toctree::
   :maxdepth: 1

   FluidFFT forge on Bitbucket <https://bitbucket.org/fluiddyn/fluidfft>
   FluidFFT in PyPI  <https://pypi.python.org/pypi/fluidfft/>
   to_do
   changes


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

