"""2d Fast Fourier Transform classes (:mod:`fluidfft.fft2d`)
============================================================

This package contains extension modules containing classes for performing Fast
Fourier Transform with different methods and libraries.

To use the FFT classes in real codes, it is simpler and recommended to use the
class :class:`fluidfft.fft2d.operators.OperatorsPseudoSpectral2D` defined in
the package

.. autosummary::
   :toctree:

   operators

All FFT classes are very similar and provide the same public functions. Since
these classes are defined in Cython extensions that can not easily be compiled
on the readthedocs server, the API of the 2d classes is presented in this fake
FFT class:

.. autoclass:: FFT2dFakeForDoc
   :members:
   :undoc-members:

"""

import os
import sys

from .. import _get_classes

__all__ = [
    "FFT2dFakeForDoc",
    "get_classes_seq",
    "get_classes_mpi",
]

if "FLUIDFFT_BUILD_DOC" in os.environ:
    from .fake_mod_fft2d_for_doc import FFT2dFakeForDoc


def get_classes_seq():
    """Return all sequential 2d classes."""
    return _get_classes(2, sequential=True)


def get_classes_mpi():
    """Return all parallel 2d classes."""
    return _get_classes(2, sequential=False)


if any("pytest" in part for part in sys.argv):
    import pytest

    pytest.register_assert_rewrite("fluidfft.fft2d.testing")
