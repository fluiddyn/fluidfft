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

from __future__ import print_function

from fluidfft._version import __version__

from importlib import import_module as _import_module

__all__ = ['__version__', 'import_fft_class', 'create_fft_object']


def import_fft_class(method, raise_import_error=True):
    """Import a fft class.

    Parameters
    ----------

    method : str
      Name of module or string characterizing a method. It has to correspond to
      a module of fluidfft. The first part "fluidfft." of the module "path" can
      be omitted.

    raise_import_error : {True}, False

      If raise_import_error == False and if there is an import error, the
      function handles the error and returns None.

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
        if raise_import_error:
            raise ImportError(method)
        else:
            print('ImportError:', method)
            return None

    return mod.FFTclass


def create_fft_object(method, n0, n1, n2=None):
    """Helper for creating fft objects.

    Parameters
    ----------

    method : str
      Name of module or string characterizing a method. It has to correspond to
      a module of fluidfft. The first part "fluidfft." of the module "path" can
      be omitted.

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
        elif (method.startswith('fft3d.') and
              method.startswith('fluidfft.fft3d.')):
            raise ValueError('Arguments incompatible')
        else:
            method = 'fluidfft.fft2d.' + method
    else:
        if method.startswith('fft3d.'):
            method = 'fluidfft.' + method
        elif method.startswith('fluidfft.fft3d.'):
            pass
        elif (method.startswith('fft2d.') and
              method.startswith('fluidfft.fft2d.')):
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
