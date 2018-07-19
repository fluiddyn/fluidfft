"""

Results are not reproducible!

$ mpirun -np 8 python investigate_small.py

--------
nb_proc: 8
N0 = 64 ; N1 = 64
Initialization (FFT2DMPIWithFFTWMPI2D) done in 2.457321 s
tests (FFT2DMPIWithFFTWMPI2D)...
 OK!

$ mpirun -np 8 python investigate_small.py

--------
nb_proc: 8
N0 = 64 ; N1 = 64
Initialization (FFT2DMPIWithFFTWMPI2D) done in 2.331123 s
tests (FFT2DMPIWithFFTWMPI2D)...
fail: (energy_X_before - energy_K_before)/energy_X_before = inf > EPS
      energy_X_before = 1.700244e-01

"""

import sys

from fluidfft.fft2d.mpi_with_fftwmpi2d import FFTclass

has_to_print = "print" in sys.argv
if has_to_print:
    from fluiddyn.util import mpi

n = 64

o = FFTclass(n, n)

if has_to_print:
    mpi.print_sorted(o.get_shapeX_loc())
    mpi.print_sorted(o.get_shapeK_loc())

o.run_tests()
