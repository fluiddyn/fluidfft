
from __future__ import print_function, division

from fluiddyn.util import mpi

from fluidfft.fft3d import get_classes_seq, get_classes_mpi

rank = mpi.rank
nb_proc = mpi.nb_proc

print_old = print


def print(*args, **kwargs):
    if mpi.rank == 0:
        print_old(*args, **kwargs)


if __name__ == '__main__':

    n = 128

    def run(FFT):
        if FFT is None:
            return
        o = FFT(n, n, n)
        o.run_tests()
        o.run_benchs()

    if nb_proc == 1 and rank == 0:
        for FFT in get_classes_seq().values():
            run(FFT)

    if nb_proc > 1:
        for FFT in get_classes_mpi().values():
            run(FFT)
