import unittest

from fluiddyn.util import mpi

from fluidfft import import_fft_class
from fluidfft.fft2d import get_classes_seq, get_classes_mpi
from fluidfft.fft2d.testing import complete_test_class_2d


def test_get_classes():
    get_classes_seq()
    get_classes_mpi()


rank = mpi.rank
nb_proc = mpi.nb_proc


methods_seq = ["pyfftw"]
methods_seq = ["fft2d.with_" + method for method in methods_seq]
classes_seq = {
    method: import_fft_class(method, raise_import_error=False)
    for method in methods_seq
}
classes_seq = {
    method: cls for method, cls in classes_seq.items() if cls is not None
}

if not classes_seq:
    raise ImportError("Not sequential 2d classes working!")


class Tests2D(unittest.TestCase):
    pass


if nb_proc == 1 and len(classes_seq) == 0:
    raise RuntimeError(
        "ImportError for all sequential classes. Nothing is working!"
    )

for method, cls in classes_seq.items():
    complete_test_class_2d(method, Tests2D, cls=cls)
