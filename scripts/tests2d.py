
from __future__ import print_function, division

from fluiddyn.util import mpi

from classes2d import classes_seq, classes_mpi

rank = mpi.rank
nb_proc = mpi.nb_proc

if __name__ == '__main__':

    n = 512

    if rank == 0:
        for cls in classes_seq:
            o = cls(n, n)
            ok = o.run_tests()

    if nb_proc > 1:
        for cls in classes_mpi:
            o = cls(n, n)
            ok = o.run_tests()
