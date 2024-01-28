import unittest
import traceback

from fluiddyn.util import mpi

from fluidfft import import_fft_class
from fluidfft.fft2d import get_classes_seq, get_classes_mpi
from fluidfft.fft2d.testing import complete_test_class_2d

try:
    import fluidfft.fft2d.with_fftw2d
except ImportError:
    # If this one does not work it is a bad sign so we want to know what happened.
    traceback.print_exc()


def test_get_classes():
    get_classes_seq()
    get_classes_mpi()


rank = mpi.rank
nb_proc = mpi.nb_proc


methods_seq = ["fftw1d", "fftw2d", "cufft", "pyfftw"]
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

if nb_proc > 1:
    methods_mpi = ["fftwmpi2d", "fftw1d"]
    methods_mpi = ["fft2d.mpi_with_" + method for method in methods_mpi]
    classes_mpi = {
        method: import_fft_class(method, raise_import_error=False)
        for method in methods_mpi
    }
    classes_mpi = {
        method: cls for method, cls in classes_mpi.items() if cls is not None
    }


class Tests2D(unittest.TestCase):
    pass


if rank == 0:
    if nb_proc == 1 and len(classes_seq) == 0:
        raise RuntimeError(
            "ImportError for all sequential classes. Nothing is working!"
        )

    for method, cls in classes_seq.items():
        complete_test_class_2d(method, Tests2D, cls=cls)


if nb_proc > 1:
    if len(classes_mpi) == 0:
        raise RuntimeError(
            "ImportError for all mpi classes. Nothing is working in mpi!"
        )

    for method, cls in classes_mpi.items():
        complete_test_class_2d(method, Tests2D, cls=cls)

# TODO: understand what was done here before!
# complete_test_class_2d("None", Tests2D, cls=False)
