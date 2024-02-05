from unittest import TestCase

from fluidfft.fft2d.testing import complete_test_class_2d


class Tests(TestCase):
    pass


for dim in "12":
    complete_test_class_2d(f"fft2d.with_fftw{dim}d", Tests)
