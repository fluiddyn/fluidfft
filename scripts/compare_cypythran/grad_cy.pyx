
cimport numpy as np
import numpy as np
np.import_array()

import cython

DTYPEf = np.float64
ctypedef np.float64_t DTYPEf_t
DTYPEc = np.complex128
ctypedef np.complex128_t DTYPEc_t


@cython.boundscheck(False)
@cython.wraparound(False)
def gradfft_from_fft_nocheck(np.ndarray[DTYPEc_t, ndim=2] f_fft,
                             np.ndarray[DTYPEf_t, ndim=2] KX,
                             np.ndarray[DTYPEf_t, ndim=2] KY):
    """Return the gradient of f_fft in spectral space."""
    cdef Py_ssize_t i0, i1, n0, n1
    cdef np.ndarray[DTYPEc_t, ndim=2] px_f_fft, py_f_fft

    n0 = f_fft.shape[0]
    n1 = f_fft.shape[1]
    
    px_f_fft = np.empty([n0, n1], dtype=np.complex128)
    py_f_fft = np.empty([n0, n1], dtype=np.complex128)

    for i0 in range(n0):
        for i1 in range(n1):
            px_f_fft[i0, i1] = 1j * KX[i0, i1] * f_fft[i0, i1]
            py_f_fft[i0, i1] = 1j * KY[i0, i1] * f_fft[i0, i1]

    return px_f_fft, py_f_fft


def gradfft_from_fft_check(np.ndarray[DTYPEc_t, ndim=2] f_fft,
                           np.ndarray[DTYPEf_t, ndim=2] KX,
                           np.ndarray[DTYPEf_t, ndim=2] KY):
    """Return the gradient of f_fft in spectral space."""
    cdef Py_ssize_t i0, i1, n0, n1
    cdef np.ndarray[DTYPEc_t, ndim=2] px_f_fft, py_f_fft

    n0 = f_fft.shape[0]
    n1 = f_fft.shape[1]
    
    px_f_fft = np.empty([n0, n1], dtype=np.complex128)
    py_f_fft = np.empty([n0, n1], dtype=np.complex128)

    for i0 in range(n0):
        for i1 in range(n1):
            px_f_fft[i0, i1] = 1j * KX[i0, i1] * f_fft[i0, i1]
            py_f_fft[i0, i1] = 1j * KY[i0, i1] * f_fft[i0, i1]

    return px_f_fft, py_f_fft
