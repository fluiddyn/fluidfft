from unittest import TestCase

from fluidfft.fft3d.testing import complete_test_class_3d
from fluidfft.fft2d.testing import complete_test_class_2d


class Tests(TestCase):
    pass


complete_test_class_2d("fft2d.with_pyfftw", Tests)
complete_test_class_3d("fft3d.with_pyfftw", Tests)
