
from fluiddyn.util import mpi

from fluidfft.fft2d import get_classes_seq, get_classes_mpi

rank = mpi.rank
nb_proc = mpi.nb_proc

print_old = print

print = mpi.printby0


if __name__ == '__main__':

    n = 1024 * 2  # / 4

    def run(FFT2D):
        if FFT2D is None:
            return
        o = FFT2D(n, n)
        o.run_tests()
        o.run_benchs()
        o.run_benchs()

    if rank == 0:
        for FFT2D in get_classes_seq().values():
            run(FFT2D)

    if nb_proc > 1:
        for FFT2D in get_classes_mpi().values():
            run(FFT2D)
