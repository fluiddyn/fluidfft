"""Class using pyfftw
=====================

.. autoclass:: FFT2DWithPYFFTW
   :members:

"""

# to get a clear ImportError in case...
import pyfftw

from fluiddyn.calcul.easypyfft import FFTW2DReal2Complex


class FFT2DWithPYFFTW(FFTW2DReal2Complex):
    def __init__(self, n0, n1):
        super(FFTclass, self).__init__(n1, n0)

    @property
    def _numpy_api(self):
        """A ``@property`` which imports and returns a NumPy-like array backend."""
        import numpy as np

        return np

    def create_arrayX(self, value=None, shape=None):
        """Return a constant array in real space."""
        return super().create_arrayX(value)

    def create_arrayK(self, value=None, shape=None):
        """Return a constant array in real space."""
        return super().create_arrayK(value)


FFTclass = FFT2DWithPYFFTW
