
from __future__ import print_function, division

from fluiddyn.util import mpi

rank = mpi.rank
nb_proc = mpi.nb_proc

from fluidfft2d.with_fftw1d import FFT2DWithFFTW1D
from fluidfft2d.with_fftw2d import FFT2DWithFFTW2D

from fluidfft2d.mpi_with_fftwmpi2d import FFT2DMPIWithFFTWMPI2D
from fluidfft2d.mpi_with_fftw1d import FFT2DMPIWithFFTW1D

classes_seq = [FFT2DWithFFTW1D, FFT2DWithFFTW2D]
classes_mpi = [FFT2DMPIWithFFTW1D, FFT2DMPIWithFFTWMPI2D]


if __name__ == '__main__':

    n = 512

    if rank == 0:
        for cls in classes_seq:
            o = cls(n, n)
            o.run_tests()

    if nb_proc > 1:
        for cls in classes_mpi:
            o = cls(n, n)
            ok = o.run_tests()
