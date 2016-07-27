"""3d Fast Fourier Transform classes
====================================

This package contains different extension modules with classes for performing
Fast Fourier Transform with different methods and libraries. The number of
classes depend on how fluidfft has been compiled.

- `fluidfft3d.with_fftw3d.FFT3DWithFFTW3D`

- `fluidfft3d.with_cufft.FFT3DWithCUFFT`

- `fluidfft3d.mpi_with_fftwmpi3d.FFT3DMPIWithFFTWMPI3D`

- `fluidfft3d.mpi_with_pfft.FFT3DMPIWithPFFT`

- `fluidfft3d.mpi_with_p3dfft.FFT3DMPIWithP3DFFT` (do not work -> need .so)

All classes are very similar and provide the same public functions. Since these
classes are defined in Cython extensions that can not easily be compiled on the
readthedocs server, the API of the 3d classes is presented in this fake FFT
class:

.. autoclass:: FFT3dFakeForDoc
   :members:
   :undoc-members:

"""

try:
    from .fake_mod_fft3d_for_doc import FFT3dFakeForDoc
except ImportError:
    pass