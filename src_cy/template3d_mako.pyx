
include 'base.pyx'


from ${module_name}_pxd cimport (
    ${class_name} as mycppclass,
    fftw_complex)


cdef class ${class_name}:
    cdef mycppclass* thisptr
    cdef tuple _shape_K_loc, _shape_X_loc

    def __cinit__(self, int n0=2, int n1=2, int n2=4):
        self.thisptr = new mycppclass(n0, n1, n2)
        
    def __init__(self, int n0=2, int n1=2, int n2=4): 
        self._shape_K_loc = self.get_local_shape_K()
        self._shape_X_loc = self.get_local_shape_X()
        
    def __dealloc__(self):
        self.thisptr.destroy()
        del self.thisptr

    def get_local_size_X(self):
        return self.thisptr.get_local_size_X()

    def run_tests(self):
        return self.thisptr.test()

    def run_benchs(self, nb_time_execute=10):
        txt = self.thisptr.bench(nb_time_execute)
        return tuple(float(word) for word in txt.split() if word[0].isdigit())

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef fft_generic(self, DTYPEf_t[:, :, ::1] fieldX,
              DTYPEc_t[:, :, ::1] fieldK=None):
        if fieldK is None:
            fieldK = np.empty(self._shape_K_loc, dtype=DTYPEc, order='C')
        self.thisptr.fft(&fieldX[0, 0, 0], <fftw_complex*> &fieldK[0, 0, 0])
        return np.array(fieldK)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef ifft_generic(self, DTYPEc_t[:, :, ::1] fieldK,
               DTYPEf_t[:, :, ::1] fieldX=None):
        if fieldX is None:
            fieldX = np.empty(self._shape_X_loc, dtype=DTYPEf, order='C')
        self.thisptr.ifft(<fftw_complex*> &fieldK[0, 0, 0], &fieldX[0, 0, 0])
        return np.array(fieldX)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef fft_as_arg(self, DTYPEf_t[:, :, ::1] fieldX,
                     DTYPEc_t[:, :, ::1] fieldK):
        self.thisptr.fft(&fieldX[0, 0, 0], <fftw_complex*> &fieldK[0, 0, 0])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef ifft_as_arg(self, DTYPEc_t[:, :, ::1] fieldK,
                      DTYPEf_t[:, :, ::1] fieldX):
        self.thisptr.ifft(<fftw_complex*> &fieldK[0, 0, 0], &fieldX[0, 0, 0])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef fft(self, DTYPEf_t[:, :, ::1] fieldX):
        cdef np.ndarray[DTYPEc_t, ndim=3] fieldK
        fieldK = np.empty(self.get_local_shape_K(), dtype=DTYPEc, order='C')
        self.thisptr.fft(&fieldX[0, 0, 0], <fftw_complex*> &fieldK[0, 0, 0])
        return fieldK

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef ifft(self, DTYPEc_t[:, :, ::1] fieldK):
        cdef np.ndarray[DTYPEf_t, ndim=3] fieldX
        fieldX = np.empty(self.get_local_shape_X(), dtype=DTYPEf, order='C')
        self.thisptr.ifft(<fftw_complex*> &fieldK[0, 0, 0], &fieldX[0, 0, 0])
        return fieldX

    cpdef get_local_shape_X(self):
        cdef int nX0loc, nX1loc, nX2
        self.thisptr.get_local_shape_X(&nX0loc, &nX1loc, &nX2)
        return nX0loc, nX1loc, nX2

    cpdef get_local_shape_K(self):
        cdef int nK0loc, nK1loc, nK2
        self.thisptr.get_local_shape_K(&nK0loc, &nK1loc, &nK2)
        return nK0loc, nK1loc, nK2

    cpdef get_global_shape_X(self):
        cdef int nX0, nX1, nX2
        self.thisptr.get_global_shape_X(&nX0, &nX1, &nX2)
        return nX0, nX1, nX2

    cpdef get_global_shape_K(self):
        cdef int nK0, nK1, nK2
        self.thisptr.get_global_shape_K(&nK0, &nK1, &nK2)
        return nK0, nK1, nK2

    cpdef sum_wavenumbers(self, np.ndarray fieldK):
        if fieldK.dtype == np.float64:
            return self.thisptr.sum_wavenumbers_double(&fieldK[0, 0, 0])
        elif fieldK.dtype == np.complex128:
            return self.thisptr.sum_wavenumbers_double(
                <fftw_complex*> &fieldK[0, 0, 0])
        else:
            raise TypeError('dtype of fieldK has to be float64 or complex128.')

    cpdef get_dimX_K(self):
        cdef int d0, d1, d2
        self.thisptr.get_dimX_K(&d0, &d1, &d2)
        return d0, d1, d2

    cdef get_global_index_first_K(self):
        cdef int i0, i1
        self.thisptr.get_global_index_first_K(&i0, &i1)
        return i0, i1
    
    cdef get_local_k_adim(self):
        int nK0, nK1, nK2, nK0_loc, nK1_loc, nK2_loc
        int d0, d1, d2, i0_start, i1_start
        np.ndarray tmp, k0_adim_loc, k1_adim_loc, k1_adim_loc

        nK0, nK1, nK2 = self.get_global_shape_K()
        nK0_loc, nK1_loc, nK2_loc = self.get_local_shape_K()
        
        d0, d1, d2 = self.get_dimX_K()
        i0_start, i1_start = self.get_global_index_first_K()

        if d0 == 2:
            tmp = np.arange(nK0seq)
        else:
            k0_adim_max = nK0//2
            tmp = np.r_[0:k0_adim_max+1, -k0_adim_max+1:0]
            
        k0_adim_loc = tmp[i0_start:i0_start+nK0_loc]

        if d1 == 2:
            tmp = np.arange(nK1seq)
        else:
            k1_adim_max = nK1//2
            tmp = np.r_[0:k1_adim_max+1, -k1_adim_max+1:0]
            
        k1_adim_loc = tmp[i1_start:i1_start+nK1_loc]
        
        if d2 == 2:
            k2_adim_loc = np.arange(nK2seq)
        else:
            k2_adim_max = nK2//2
            k2_adim_loc = np.r_[0:k2_adim_max+1, -k2_adim_max+1:0]
            
        return k0_adim_loc, k1_adim_loc, k2_adim_loc
