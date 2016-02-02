
include 'base.pyx'


from ${module_name}_pxd cimport (
    ${class_name} as mycppclass,
    mycomplex)


cdef class ${class_name}:
    cdef mycppclass* thisptr
    cdef tuple _shape_K_loc, _shape_X_loc

    def __cinit__(self, int n0=2, int n1=2):
        self.thisptr = new mycppclass(n0, n1)
        
    def __init__(self, int n0=2, int n1=2): 
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
    cpdef fft(self, DTYPEf_t[:, ::1] fieldX,
              DTYPEc_t[:, ::1] fieldK=None):
        if fieldK is None:
            fieldK = np.empty(self._shape_K_loc, dtype=DTYPEc, order='C')
        self.thisptr.fft(&fieldX[0, 0], <mycomplex*> &fieldK[0, 0])
        return np.array(fieldK)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef ifft(self, DTYPEc_t[:, ::1] fieldK,
               DTYPEf_t[:, ::1] fieldX=None):
        if fieldX is None:
            fieldX = np.empty(self._shape_X_loc, dtype=DTYPEf, order='C')
        self.thisptr.ifft(<mycomplex*> &fieldK[0, 0], &fieldX[0, 0])
        return np.array(fieldX)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef fft_as_arg(self, DTYPEf_t[:, ::1] fieldX,
                     DTYPEc_t[:, ::1] fieldK):
        self.thisptr.fft(&fieldX[0, 0], <mycomplex*> &fieldK[0, 0])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef ifft_as_arg(self, DTYPEc_t[:, ::1] fieldK,
                      DTYPEf_t[:, ::1] fieldX):
        self.thisptr.ifft(<mycomplex*> &fieldK[0, 0], &fieldX[0, 0])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef return_fft(self, DTYPEf_t[:, ::1] fieldX):
        cdef np.ndarray[DTYPEc_t, ndim=2] fieldK
        fieldK = np.empty(self.get_local_shape_K(), dtype=DTYPEc, order='C')
        self.thisptr.fft(&fieldX[0, 0], <mycomplex*> &fieldK[0, 0])
        return fieldK

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef return_ifft(self, DTYPEc_t[:, ::1] fieldK):
        cdef np.ndarray[DTYPEf_t, ndim=2] fieldX
        fieldX = np.empty(self.get_local_shape_X(), dtype=DTYPEf, order='C')
        self.thisptr.ifft(<mycomplex*> &fieldK[0, 0], &fieldX[0, 0])
        return fieldX

    cpdef get_local_shape_X(self):
        cdef int output_nX0loc, output_nX1
        self.thisptr.get_local_shape_X(&output_nX0loc, &output_nX1)
        return output_nX0loc, output_nX1

    cpdef get_local_shape_K(self):
        cdef int output_nK0loc, output_nK1
        self.thisptr.get_local_shape_K(&output_nK0loc, &output_nK1)
        return output_nK0loc, output_nK1
