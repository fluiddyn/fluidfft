
from __future__ import print_function, division

import unittest

import numpy as np

from fluiddyn.util import mpi

from fluidfft.fft2d import get_classes_seq, get_classes_mpi

n = 128

rank = mpi.rank
nb_proc = mpi.nb_proc

classes = get_classes_seq()

if nb_proc > 1:
    classes.update(get_classes_mpi())


def make_test_function(cls):

    def test(self):
        o = cls(n, n)
        o.run_tests()

        a = np.random.random(o.get_local_size_X()).reshape(
            o.get_shapeX_loc())
        afft = o.fft(a)
        a = o.ifft(afft)
        afft = o.fft(a)

        EX = o.compute_energy_from_X(a)
        EK = o.compute_energy_from_K(afft)

        self.assertTrue(EX == EK and EX != 0.)

        k0, k1 = o.get_k_adim_loc()

        if o.get_shapeK_loc() != (k0.size, k1.size):
            print(o.get_shapeK_loc(), k0.size, k1.size)
            raise Exception('o.get_shapeK_loc() != (k0.size, k1.size)')

    return test


class Tests2D(unittest.TestCase):
    pass


for name, cls in classes.items():
    if cls is None:
        continue

    setattr(Tests2D, 'test_{0}'.format(name), make_test_function(cls))


if __name__ == '__main__':
    unittest.main()
