from unittest import TestCase

from fluidfft.fft3d.testing import complete_test_class_3d


class Tests(TestCase):
    pass


complete_test_class_3d("fft3d.mpi_with_p3dfft", Tests)
