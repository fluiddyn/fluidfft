from unittest import TestCase

from fluidfft.fft3d.testing import complete_test_class_3d


class Tests(TestCase):
    pass


# Importing pfft can fail because of CPU incompatibility
complete_test_class_3d("fft3d.mpi_with_pfft", Tests, skip_if_import_error=False)
