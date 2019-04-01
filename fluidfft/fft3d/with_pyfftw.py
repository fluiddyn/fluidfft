"""Class using pyfftw (:mod:`fluidfft.fft3d.with_pyfftw`)
=========================================================

.. autoclass:: FFT3DWithPYFFTW
   :members:

"""

# to get a clear ImportError in case...
import pyfftw

from fluiddyn.calcul.easypyfft import FFTW3DReal2Complex


class FFT3DWithPYFFTW(FFTW3DReal2Complex):
    def __init__(self, n0, n1, n2):
        super(FFTclass, self).__init__(n2, n1, n0)

    def gather_Xspace(self, arr, root=None):
        return arr

    def scatter_Xspace(self, arr, root=None):
        return arr

    @property
    def _numpy_api(self):
        import numpy as np
        return np


FFTclass = FFT3DWithPYFFTW
