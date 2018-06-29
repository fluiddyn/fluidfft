# cython: embedsignature=True
# cython: language_level=3

# DEF MPI4PY = 0

cimport cython

from cython cimport view

cimport numpy as np
import numpy as np
np.import_array()

try:
    from mpi4py import MPI
except ImportError:
    nb_proc = 1
    rank = 0
else:
    comm = MPI.COMM_WORLD
    nb_proc = comm.size
    rank = comm.Get_rank()

IF MPI4PY:
    from mpi4py cimport MPI

    #from mpi4py.mpi_c cimport *
    # for mpi4py > 2.0
    from mpi4py.libmpi cimport *

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

# workaround to avoid cython bug https://github.com/cython/cython/issues/2093
# contiguous = 1
contiguous = view.contiguous

ctypedef DTYPEf_t[:, :, ::contiguous] view3df_t
ctypedef DTYPEc_t[:, :, ::contiguous] view3dc_t
ctypedef DTYPEf_t[:, ::contiguous] view2df_t
ctypedef DTYPEc_t[:, ::contiguous] view2dc_t

include 'util_pyfftw.pyx'
