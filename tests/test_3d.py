import unittest
import traceback

import numpy as np

from fluiddyn.util import mpi

from fluidfft.fft3d import get_classes_seq, get_classes_mpi
from fluidfft.fft3d.testing import make_testop_functions

try:
    import fluidfft.fft3d.with_fftw3d
except ImportError:
    # If this one does not work it is a bad sign so we want to know what appends.
    traceback.print_exc()


n = 8

rank = mpi.rank
nb_proc = mpi.nb_proc

classes_seq = get_classes_seq()
classes_seq = {name: cls for name, cls in classes_seq.items() if cls is not None}

if not classes_seq:
    raise ImportError("Not sequential 2d classes working!")

if nb_proc > 1:
    classes_mpi = get_classes_mpi()
    classes_mpi = {
        name: cls for name, cls in classes_mpi.items() if cls is not None
    }


class Tests3D(unittest.TestCase):
    pass


def complete_class(name, cls):
    tests = make_testop_functions(name, cls)

    for key, test in tests.items():
        setattr(Tests3D, "test_operator3d_{}_{}".format(name, key), test)


if nb_proc == 1:
    if len(classes_seq) == 0:
        raise Exception(
            "ImportError for all sequential classes. Nothing is working!"
        )

    for name, cls in classes_seq.items():
        complete_class(name, cls)

    complete_class("None", None)


if nb_proc > 1:
    if len(classes_mpi) == 0:
        raise Exception(
            "ImportError for all mpi classes. Nothing is working in mpi!"
        )

    for name, cls in classes_mpi.items():
        complete_class(name, cls)
