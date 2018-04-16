"""Class using pyfftw (:mod:`fluidfft.fft3d.with_pyfftw`)
=========================================================

.. autoclass:: FFT3DWithPYFFTW
   :members:

"""

from warnings import warn
from time import time

import numpy as np

from fluiddyn.calcul.easypyfft import FFTW3DReal2Complex


class FFT3DWithPYFFTW(FFTW3DReal2Complex):

    def __init__(self, n0, n1, n2):
        super(FFTclass, self).__init__(n2, n1, n0)

    def gather_Xspace(self, arr, root=None):
        return arr

    def scatter_Xspace(self, arr, root=None):
        return arr


FFTclass = FFT3DWithPYFFTW
