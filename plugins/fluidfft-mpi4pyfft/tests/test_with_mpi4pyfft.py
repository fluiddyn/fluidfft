from unittest import TestCase

from fluidfft.fft3d.testing import complete_test_class_3d


class Tests(TestCase):
    pass


methods = ["fft3d.mpi_with_mpi4pyfft", "fft3d.mpi_with_mpi4pyfft_slab"]
for method in methods:
    complete_test_class_3d(method, Tests)
