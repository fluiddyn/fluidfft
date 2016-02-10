"""Simple Fast Fourier Transform for Python
===========================================

This module provides two helper functions to import fft classes and
create fft objects:

.. autofunction:: import_fft_class

.. autofunction:: create_fft_object

The fft classes are in the two subpackages

.. autosummary::
   :toctree:

   fft2d
   fft3d

"""

from fluidfft._version import __version__

from importlib import import_module as _import_module


def import_fft_class(method):
    """Import a fft class.

    Parameters
    ----------

    method : str
      Name of module or string characterizing a method.

    Returns
    -------

    The corresponding FFT class.

    """
    if method.startswith('fft2d.') or method.startswith('fft3d.'):
        method = 'fluidfft.' + method
    elif method.startswith('fluidfft.fft2d.'):
        pass
    else:
        raise ValueError

    try:
        mod = _import_module(method)
    except ImportError:
        raise ImportError(method)

    return mod.FFTclass


def create_fft_object(method, n0, n1, n2=None):
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

    if n2 is None:
        if method.startswith('fft2d.'):
            method = 'fluidfft.' + method
        elif method.startswith('fluidfft.fft2d.'):
            pass
        elif method.startswith('fft3d.') and \
             method.startswith('fluidfft.fft3d.'):
            raise ValueError('Arguments incompatible')
        else:
            method = 'fluidfft.fft2d.' + method
    else:
        if method.startswith('fft3d.'):
            method = 'fluidfft.' + method
        elif method.startswith('fluidfft.fft3d.'):
            pass
        elif method.startswith('fft2d.') and \
             method.startswith('fluidfft.fft2d.'):
            raise ValueError('Arguments incompatible')
        else:
            method = 'fluidfft.fft3d.' + method

    try:
        mod = _import_module(method)
    except ImportError:
        raise ImportError(method)

    if n2 is None:
        return mod.FFTclass(n0, n1)
    else:
        return mod.FFTclass(n0, n1, n2)
