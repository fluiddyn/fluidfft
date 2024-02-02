from mpi4py cimport MPI
from mpi4py.libmpi cimport *

# fix a bug arising when using a recent version of mpi4py
cdef extern from 'mpi-compat.h': pass
