from unittest import TestCase

from fluidfft.fft2d.testing import complete_test_class_2d


class Tests(TestCase):
    pass


complete_test_class_2d("fft2d.mpi_with_fftw1d", Tests)
