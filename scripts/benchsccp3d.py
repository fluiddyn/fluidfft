
from __future__ import print_function, division

from fluiddyn.util import mpi
rank = mpi.rank
nb_proc = mpi.nb_proc

from classes3d import classes_seq, classes_mpi

print_old = print


def print(*args, **kwargs):
    if mpi.rank == 0:
        print_old(*args, **kwargs)


if __name__ == '__main__':

    n = 128

    def run(FFT2D):
        o = FFT2D(n, n, n)
        o.run_tests()
        o.run_benchs()

    if rank == 0:
        for FFT2D in classes_seq:
            run(FFT2D)

    if nb_proc > 1:
        for FFT2D in classes_mpi:
            run(FFT2D)
