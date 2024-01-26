import unittest

try:
    import pyfftw
except ImportError:
    pyfftw = False

try:
    from fluidfft.fft2d import with_fftw2d
except ImportError:
    with_fftw2d = False

try:
    from fluidfft.fft3d import with_fftw3d
except ImportError:
    with_fftw3d = False


from fluiddyn.util.mpi import rank

from fluidfft import create_fft_object


class TestsCreateFFTObject(unittest.TestCase):
    def test2d(self):
        if rank > 0:
            return

        if pyfftw:
            method = "fft2d.with_pyfftw"
        elif with_fftw2d:
            method = "fft2d.with_fftw2d"
        else:
            raise ImportError("No simple 2d classes work.")

        create_fft_object(method, 4, 4)

    def test3d(self):
        if rank > 0:
            return

        if pyfftw:
            method = "fft3d.with_pyfftw"
        elif with_fftw3d:
            method = "fft3d.with_fftw3d"
        else:
            raise ImportError("No simple 3d classes work.")

        create_fft_object(method, 4, 4, 4)
