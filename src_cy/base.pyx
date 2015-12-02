

cimport cython

cimport numpy as np
import numpy as np
np.import_array()

from mpi4py import MPI
from mpi4py cimport MPI

# fix a bug arising when using a recent version of mpi4py
cdef extern from 'mpi-compat.h': pass

# we define python and c types for physical and Fourier spaces
DTYPEb = np.uint8
ctypedef np.uint8_t DTYPEb_t
DTYPEi = np.int
ctypedef np.int_t DTYPEi_t
DTYPEf = np.float64
ctypedef np.float64_t DTYPEf_t
DTYPEc = np.complex128
ctypedef np.complex128_t DTYPEc_t
