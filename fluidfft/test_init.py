
from __future__ import print_function, division

import unittest

from fluiddyn.util.mpi import rank

from . import create_fft_object



class TestsCreateFFTObject(unittest.TestCase):

    def test2d(self):
        if rank == 0:
            create_fft_object('fft2d.with_fftw2d', 4, 4)

    def test3d(self):
        if rank == 0:
            create_fft_object('fft3d.with_fftw3d', 4, 4, 4)
