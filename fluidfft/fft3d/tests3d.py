
from __future__ import print_function, division

import unittest

from fluiddyn.util import mpi

from fluidfft.fft3d import get_classes_seq, get_classes_mpi

n = 16

rank = mpi.rank
nb_proc = mpi.nb_proc


def make_test_function(cls, sequencial):

    def test(self):
        if sequencial and rank > 0:
            return
        o = cls(n, n)
        o.run_tests()

    return test


class Tests3D(unittest.TestCase):
    pass


for name, cls in get_classes_seq().items():
    if cls is None:
        continue
    setattr(Tests3D, 'test_{0}'.format(name), make_test_function(cls, True))

for name, cls in get_classes_mpi().items():
    if cls is None:
        continue
    setattr(Tests3D, 'test_{0}'.format(name), make_test_function(cls, False))


if __name__ == '__main__':
    unittest.main()
