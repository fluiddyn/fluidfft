import unittest
import traceback

from fluiddyn.util import mpi

from fluidfft import import_fft_class
from fluidfft.fft3d import get_classes_seq, get_classes_mpi
from fluidfft.fft3d.testing import complete_test_class_3d

try:
    import fluidfft_fftw.fft3d.with_fftw3d
except ImportError:
    # If this one does not work it is a bad sign so we want to know what appends.
    traceback.print_exc()


def test_get_classes():
    get_classes_seq()
    get_classes_mpi()


methods_seq = ["pyfftw"]
methods_seq = ["fft3d.with_" + method for method in methods_seq]
classes_seq = {
    method: import_fft_class(method, raise_import_error=False)
    for method in methods_seq
}
classes_seq = {
    method: cls for method, cls in classes_seq.items() if cls is not None
}
if not classes_seq:
    raise ImportError("Not sequential 2d classes working!")


class Tests3D(unittest.TestCase):
    pass


if mpi.nb_proc == 1:
    if len(classes_seq) == 0:
        raise RuntimeError(
            "ImportError for all sequential classes. Nothing is working!"
        )

    for method, cls in classes_seq.items():
        complete_test_class_3d(method, Tests3D, cls=cls)

    # TODO: understand what was done here before!
    # complete_class("None", None)
