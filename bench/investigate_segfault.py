"""
Segfault:

mpirun -np 16 python investigate_segfault.py

mpirun -np 16 python investigate_segfault.py print

No segfault:

mpirun -np 8 python investigate_segfault.py

"""

import sys

from fluidfft.fft2d.mpi_with_fftwmpi2d import FFTclass

has_to_print = "print" in sys.argv
if has_to_print:
    from fluiddyn.util import mpi

n = 2048

o = FFTclass(n, n)

if has_to_print:
    mpi.print_sorted(o.get_shapeX_loc())
    mpi.print_sorted(o.get_shapeK_loc())

o.run_tests()
o.run_benchs(1)
