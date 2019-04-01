"""Class using pyfftw (:mod:`fluidfft.fft2d.with_pyfftw`)
=========================================================

.. autoclass:: FFT2DWithPYFFTW
   :members:

"""

# to get a clear ImportError in case...
import numpy as np
import pyfftw

from fluiddyn.calcul.easypyfft import FFTW2DReal2Complex


class FFT2DWithPYFFTW(FFTW2DReal2Complex):
    def __init__(self, n0, n1):
        super(FFTclass, self).__init__(n1, n0)
        self.empty_aligned = pyfftw.empty_aligned
        self.byte_align = pyfftw.byte_align

    @property
    def _numpy_api(self):
        """A ``@property`` which imports and returns a NumPy-like array backend."""
        import numpy as np
        return np

    def create_arrayX(self, value=None, shape=None):
        """Return a constant array in real space."""
        field = pyfftw.empty_aligned(shape)
        if value is not None:
            field.fill(value)
        return field

    def create_arrayK(self, value=None, shape=None):
        """Return a constant array in real space."""
        field = pyfftw.empty_aligned(shape, dtype=np.complex128)
        if value is not None:
            field.fill(value)
        return field


FFTclass = FFT2DWithPYFFTW
