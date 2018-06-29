"""Class using pyfftw (:mod:`fluidfft.fft2d.with_pyfftw`)
=========================================================

.. autoclass:: FFT2DWithPYFFTW
   :members:

"""

# to get a clear ImportError in case...
import pyfftw

from fluiddyn.calcul.easypyfft import FFTW2DReal2Complex


class FFT2DWithPYFFTW(FFTW2DReal2Complex):
    def __init__(self, n0, n1):
        super(FFTclass, self).__init__(n1, n0)


FFTclass = FFT2DWithPYFFTW
