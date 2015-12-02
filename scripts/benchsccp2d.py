
from __future__ import print_function, division

from time import time

import numpy as np

from fluiddyn.util import mpi
rank = mpi.rank
nb_proc = mpi.nb_proc

from fluidfft2d.with_fftw1d import FFT2DWithFFTW1D
from fluidfft2d.with_fftw2d import FFT2DWithFFTW2D

from fluidfft2d.mpi_with_fftwmpi2d import FFT2DMPIWithFFTWMPI2D
from fluidfft2d.mpi_with_fftw1d import FFT2DMPIWithFFTW1D


classes_seq = [FFT2DWithFFTW1D, FFT2DWithFFTW2D]
classes_mpi = [FFT2DMPIWithFFTW1D, FFT2DMPIWithFFTWMPI2D]

classes = classes_seq + classes_mpi

print_old = print


def print(*args, **kwargs):
    if mpi.rank == 0:
        print_old(*args, **kwargs)

if __name__ == '__main__':

    n = 1024 * 2  # / 4

    def run(FFT2D):
        o = FFT2D(n, n)
        o.run_tests()
        o.run_benchs()
        o.run_benchs()

    if rank == 0:
        for FFT2D in classes_seq:
            run(FFT2D)

    if nb_proc > 1:
        for FFT2D in classes_mpi:
            run(FFT2D)
