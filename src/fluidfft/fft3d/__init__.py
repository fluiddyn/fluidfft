"""3d Fast Fourier Transform classes (:mod:`fluidfft.fft3d`)
============================================================

This package contains extension modules containing classes for performing Fast
Fourier Transform with different methods and libraries.

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

import os
import sys

from .. import _get_classes

__all__ = [
    "FFT3dFakeForDoc",
    "get_classes_seq",
    "get_classes_mpi",
]

if "FLUIDFFT_BUILD_DOC" in os.environ:
    from .fake_mod_fft3d_for_doc import FFT3dFakeForDoc


def get_classes_seq():
    """Return all sequential 3d classes."""
    return _get_classes(3, sequential=True)


def get_classes_mpi():
    """Return all parallel 3d classes."""
    return _get_classes(3, sequential=False)


if any("pytest" in part for part in sys.argv):
    import pytest

    pytest.register_assert_rewrite("fluidfft.fft3d.testing")
