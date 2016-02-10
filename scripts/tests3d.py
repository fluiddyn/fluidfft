
from __future__ import print_function, division

from fluiddyn.util import mpi

rank = mpi.rank
nb_proc = mpi.nb_proc

from classes3d import classes_seq, classes_mpi


if __name__ == '__main__':

    n = 16

    if rank == 0:
        for cls in classes_seq:
            o = cls(n, 2*n, n)
            o.run_tests()

    if nb_proc > 1:
        for cls in classes_mpi:
            o = cls(n, 2*n, n)
            ok = o.run_tests()
