
from __future__ import print_function, division

import numpy as np

from fluiddyn.util import mpi

from classes2d import classes_seq, classes_mpi

rank = mpi.rank
nb_proc = mpi.nb_proc


def test(cls, n):
    o = cls(n, n)
    o.run_tests()

    a = np.random.random(o.get_local_size_X()).reshape(o.get_shapeX_loc())
    afft = o.fft(a)
    a = o.ifft(afft)
    afft = o.fft(a)

    EX = o.compute_energy_from_X(a)
    EK = o.compute_energy_from_K(afft)

    assert EX == EK and EX != 0.

    k0, k1 = o.get_k_adim_loc()

    if o.get_shapeK_loc() != (k0.size, k1.size):
        print(o.get_shapeK_loc(), k0.size, k1.size)
        raise Exception('o.get_shapeK_loc() != (k0.size, k1.size)')


if __name__ == '__main__':

    n = 512

    if rank == 0:
        for cls in classes_seq:
            test(cls, n)

    if nb_proc > 1:
        for cls in classes_mpi:
            test(cls, n)
