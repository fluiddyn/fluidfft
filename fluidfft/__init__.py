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

from fluiddyn.util.mpi import printby0

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
    if method == 'sequential':
        method = 'fft2d.with_fftw2d'

    if method.startswith('fft2d.') or method.startswith('fft3d.'):
        method = 'fluidfft.' + method
    else:
        raise ValueError(
            "not (method.startswith('fft2d.') or method.startswith('fft3d.'))")

    try:
        mod = _import_module(method)
    except ImportError:
        if raise_import_error:
            raise ImportError(method)
        else:
            printby0('ImportError:', method)
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

    cls = import_fft_class(method)

    str_module = cls.__module__

    if n2 is None and str_module.startswith('fluidfft.fft3d.'):
        raise ValueError('Arguments incompatible')
    elif n2 is not None and str_module.startswith('fluidfft.fft2d.'):
        raise ValueError('Arguments incompatible')

    if n2 is None:
        return cls(n0, n1)
    else:
        return cls(n0, n1, n2)
