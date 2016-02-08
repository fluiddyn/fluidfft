"""2d Fast Fourier Transform classes
====================================

This package contains different extension modules with classes for performing
Fast Fourier Transform with different methods and libraries. The number of
classes depend on how fluidfft has been compiled.

- `fluidfft2d.with_fftw1d.FFT2DWithFFTW1D`

- `fluidfft2d.with_fftw1d.FFT2DWithFFTW2D`

- `fluidfft2d.with_cufft.FFT2DWithCUFFT`

- `fluidfft2d.mpi_with_fftwmpi2d.FFT2DMPIWithFFTW1D`

- `fluidfft2d.mpi_with_fftwmpi2d.FFT2DMPIWithFFTWMPI2D`

The module also defines a helper function to create the fft objects:

.. autofunction:: create_fft_object

All classes are very similar and provide the same public functions. Since these
classes are defined in Cython extensions that can not easily be compiled on the
readthedocs server, the API of the 2d classes is presented in this fake FFT
class:

.. autoclass:: FFT2dFakeForDoc
   :members:
   :undoc-members:

"""


from ._version import __version__


def create_fft_object(method, n0, n1, n2):
    """Helper for creating fft objects.

    Parameters
    ----------

    method : str
      Name of module or string characterizing a method.

    n0, n1, n2 : int
      Dimensions of the real space array (in sequential).

    Returns
    -------

    The corresponding FFT object.


    """

    raise NotImplementedError


try:
    from .fake_mod_fft2d_for_doc import FFT2dFakeForDoc
except ImportError:
    pass
