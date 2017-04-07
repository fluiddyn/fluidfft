"""2d Fast Fourier Transform classes
====================================

This package contains different extension modules with classes for performing
Fast Fourier Transform with different methods and libraries. The number of
classes depend on how fluidfft has been compiled.

- :class:`fluidfft.fft2d.with_fftw1d.FFT2DWithFFTW1D`

- :class:`fluidfft.fft2d.with_fftw2d.FFT2DWithFFTW2D`

- :class:`fluidfft.fft2d.with_cufft.FFT2DWithCUFFT`

- :class:`fluidfft.fft2d.mpi_with_fftwmpi2d.FFT2DMPIWithFFTW1D`

- :class:`fluidfft.fft2d.mpi_with_fftwmpi2d.FFT2DMPIWithFFTWMPI2D`

All classes are very similar and provide the same public functions. Since these
classes are defined in Cython extensions that can not easily be compiled on the
readthedocs server, the API of the 2d classes is presented in this fake FFT
class:

.. autoclass:: FFT2dFakeForDoc
   :members:
   :undoc-members:

"""

from .. import import_fft_class

__all__ = ['FFT2dFakeForDoc', 'methods_seq', 'methods_mpi',
           'get_classes_seq', 'get_classes_mpi']

try:
    from .fake_mod_fft2d_for_doc import FFT2dFakeForDoc
except ImportError:
    pass

methods_seq = ['fftw1d', 'fftw2d', 'cufft']
methods_seq = ['fft2d.with_' + method for method in methods_seq]

methods_mpi = ['fftwmpi2d', 'fftw1d']
methods_mpi = ['fft2d.mpi_with_' + method for method in methods_mpi]


def get_classes_seq():
    return {method: import_fft_class(method, raise_import_error=False)
            for method in methods_seq}


def get_classes_mpi():
    return {method: import_fft_class(method, raise_import_error=False)
            for method in methods_mpi}
