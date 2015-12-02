
from __future__ import print_function, division

from time import time

import numpy as np

from fluiddyn.util import mpi
rank = mpi.rank
nb_proc = mpi.nb_proc

from fluidfft3d.with_fftw3d import FFT3DWithFFTW3D

from fluidfft3d.mpi_with_fftwmpi3d import FFT3DMPIWithFFTWMPI3D
from fluidfft3d.mpi_with_pfft import FFT3DMPIWithPFFT

classes_seq = [FFT3DWithFFTW3D]
classes_mpi = [FFT3DMPIWithFFTWMPI3D, FFT3DMPIWithPFFT]


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
