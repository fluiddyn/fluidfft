"""3d Fast Fourier Transform classes (:mod:`fluidfft.fft3d`)
============================================================

This package contains extension modules containing classes for performing Fast
Fourier Transform with different methods and libraries. The number of classes
depend on how fluidfft has been compiled. The 3d classes currently implemented
are:

- :class:`fluidfft3d.with_fftw3d.FFT3DWithFFTW3D`
- :class:`fluidfft3d.with_cufft.FFT3DWithCUFFT`
- :class:`fluidfft3d.mpi_with_fftwmpi3d.FFT3DMPIWithFFTWMPI3D`
- :class:`fluidfft3d.mpi_with_fftwmpi3d.FFT3DMPIWithFFTW1D`
- :class:`fluidfft3d.mpi_with_pfft.FFT3DMPIWithPFFT`
- :class:`fluidfft3d.mpi_with_p3dfft.FFT3DMPIWithP3DFFT`
- :class:`fluidfft3d.mpi_with_mpi4pyfft.FFT3DMPIWithMPI4PYFFT`
- :class:`fluidfft3d.mpi_with_mpi4pyfft_slab.FFT3DMPIWithMPI4PYFFTSlab`

To use the FFT classes in real codes, it is simpler and recommended to use the
class :class:`fluidfft.fft3d.operators.OperatorsPseudoSpectral3D` defined in
the package

.. autosummary::
   :toctree:

   operators

All FFT classes are very similar and provide the same public functions. Since
these classes are defined in Cython extensions that can not easily be compiled
on the readthedocs server, the API of the 3d classes is presented in this fake
FFT class:

.. autoclass:: FFT3dFakeForDoc
   :members:
   :undoc-members:

"""

from .. import import_fft_class

__all__ = [
    "FFT3dFakeForDoc",
    "methods_seq",
    "methods_mpi",
    "get_classes_seq",
    "get_classes_mpi",
]

try:
    from .fake_mod_fft3d_for_doc import FFT3dFakeForDoc
except ImportError:
    pass

methods_seq = ["fftw3d", "pyfftw"]
methods_seq = ["fft3d.with_" + method for method in methods_seq]

methods_mpi = [
    "fftw1d",
    "fftwmpi3d",
    "p3dfft",
    "pfft",
    "mpi4pyfft",
    "mpi4pyfft_slab",
]
methods_mpi = ["fft3d.mpi_with_" + method for method in methods_mpi]


def get_classes_seq():
    """Return all sequential 3d classes."""
    return {
        method: import_fft_class(method, raise_import_error=False)
        for method in methods_seq
    }


def get_classes_mpi():
    """Return all parallel 3d classes."""
    return {
        method: import_fft_class(method, raise_import_error=False)
        for method in methods_mpi
    }
