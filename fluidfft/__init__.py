"""2d Fast Fourier Transform classes
====================================

This module provides a helper function to create the fft objects:

.. autofunction:: create_fft_object

"""

from fluidfft._version import __version__

from importlib import import_module as _import_module

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
        if method.startswith('2d.'):
            method = 'fluidfft' + method
        elif method.startswith('fluid2d.'):
            pass
        elif method.startswith('3d.') and method.startswith('fluidfft3d.'):
            raise ValueError('Arguments incompatible')
        else:
            method = 'fluidfft2d.' + method
    else:
        if method.startswith('3d.'):
            method = 'fluidfft' + method
        elif method.startswith('fluidfft3d.'):
            pass
        elif method.startswith('2d.') and method.startswith('fluidfft2d.'):
            raise ValueError('Arguments incompatible')
        else:
            method = 'fluidfft3d.' + method

    try:
        mod = _import_module(method)
    except ImportError:
        raise ImportError(method)

    if n2 is None:
        return mod.FFTclass(n0, n1)
    else:
        return mod.FFTclass(n0, n1, n2)
