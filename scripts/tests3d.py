
from __future__ import print_function, division

from fluiddyn.util import mpi

rank = mpi.rank
nb_proc = mpi.nb_proc

from fluidfft3d.with_fftw3d import FFT3DWithFFTW3D
from fluidfft3d.with_cufft import FFT3DWithCUFFT

from fluidfft3d.mpi_with_fftwmpi3d import FFT3DMPIWithFFTWMPI3D
from fluidfft3d.mpi_with_pfft import FFT3DMPIWithPFFT

classes_seq = [FFT3DWithFFTW3D, FFT3DWithCUFFT]
classes_mpi = [FFT3DMPIWithFFTWMPI3D, FFT3DMPIWithPFFT]


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
