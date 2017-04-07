
include 'base.pyx'


from ${module_name}_pxd cimport (
    ${class_name} as mycppclass,
    mycomplex)


cdef class ${class_name}:
    """Class to perform Fast Fourier Transform in 2d."""
    cdef mycppclass* thisptr
    cdef tuple _shape_K_loc, _shape_X_loc

    def __cinit__(self, int n0=2, int n1=2):
        self.thisptr = new mycppclass(n0, n1)

    def __init__(self, int n0=2, int n1=2):
        self._shape_K_loc = self.get_shapeK_loc()
        self._shape_X_loc = self.get_shapeX_loc()

    def __dealloc__(self):
        self.thisptr.destroy()
        del self.thisptr

    def get_local_size_X(self):
        return self.thisptr.get_local_size_X()
    
    def run_tests(self):
        """Run the c++ tests."""
        return self.thisptr.test()

    def run_benchs(self, nb_time_execute=10):
        """Run the c++ benchmarcks"""
        txt = self.thisptr.bench(nb_time_execute)
        return tuple(float(word) for word in txt.split() if word[0].isdigit())

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef fft_as_arg(self, DTYPEf_t[:, ::1] fieldX,
                     DTYPEc_t[:, ::1] fieldK):
        """Perform the fft and copy the result in the second argument."""
        self.thisptr.fft(&fieldX[0, 0], <mycomplex*> &fieldK[0, 0])
        return fieldK
        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef ifft_as_arg(self, DTYPEc_t[:, ::1] fieldK,
                      DTYPEf_t[:, ::1] fieldX):
        """Perform the ifft and copy the result in the second argument."""
        self.thisptr.ifft(<mycomplex*> &fieldK[0, 0], &fieldX[0, 0])
        return fieldX

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef fft(self, DTYPEf_t[:, ::1] fieldX):
        """Perform the fft and returns the result."""
        cdef np.ndarray[DTYPEc_t, ndim=2] fieldK
        fieldK = np.empty(self.get_shapeK_loc(), dtype=DTYPEc, order='C')
        self.thisptr.fft(&fieldX[0, 0], <mycomplex*> &fieldK[0, 0])
        return fieldK

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef ifft(self, DTYPEc_t[:, ::1] fieldK):
        """Perform the ifft and returns the result."""
        cdef np.ndarray[DTYPEf_t, ndim=2] fieldX
        fieldX = np.empty(self.get_shapeX_loc(), dtype=DTYPEf, order='C')
        self.thisptr.ifft(<mycomplex*> &fieldK[0, 0], &fieldX[0, 0])
        return fieldX

    def get_shapeX_loc(self):
        cdef int output_nX0loc, output_nX1
        self.thisptr.get_local_shape_X(&output_nX0loc, &output_nX1)
        return output_nX0loc, output_nX1

    def get_shapeK_loc(self):
        cdef int output_nK0loc, output_nK1
        self.thisptr.get_local_shape_K(&output_nK0loc, &output_nK1)
        return output_nK0loc, output_nK1

    def get_shapeX_seq(self):
        cdef int output_nX0, output_nX1
        self.thisptr.get_shapeX_seq(&output_nX0, &output_nX1)
        return output_nX0, output_nX1

    def get_shapeK_seq(self):
        cdef int output_nK0, output_nK1
        self.thisptr.get_shapeK_seq(&output_nK0, &output_nK1)
        return output_nK0, output_nK1

    def get_is_transposed(self):
        """Get the boolean "is_transposed"."""
        return bool(self.thisptr.get_is_transposed())

    def get_seq_indices_first_X(self):
        return <int> self.thisptr.get_local_X0_start(), 0

    def get_seq_indices_first_K(self):
        return <int> self.thisptr.get_local_K0_start(), 0

    def get_k_adim_loc(self):
        nyseq, nxseq = self.get_shapeX_seq()

        kyseq = np.array(list(range(nyseq//2 + 1)) + list(range(-nyseq//2 + 1, 0)))
        kxseq = np.array(range(nxseq//2 + 1))

        if self.get_is_transposed():
            k0seq, k1seq = kxseq, kyseq
        else:
            k0seq, k1seq = kyseq, kxseq

        ik0_start, ik1_start = self.get_seq_indices_first_K()
        nk0loc, nk1loc = self.get_shapeK_loc()

        k0loc = k0seq[ik0_start:ik0_start+nk0loc]
        k1loc = k1seq[ik1_start:ik1_start+nk1loc]

        return k0loc, k1loc

    def get_x_adim_loc(self):
        nyseq, nxseq = self.get_shapeX_seq()

        ix0_start, ix1_start = self.get_seq_indices_first_X()
        nx0loc, nx1loc = self.get_shapeX_loc()

        x0loc = np.array(range(ix0_start, ix0_start+nx0loc))
        x1loc = np.array(range(ix1_start, ix1_start+nx1loc))

        return x0loc, x1loc

    def compute_energy_from_X(self, DTYPEf_t[:, ::1] fieldX):
        return <float> self.thisptr.compute_energy_from_X(&fieldX[0, 0])

    def compute_energy_from_K(self, DTYPEc_t[:, ::1] fieldK):
        return <float> self.thisptr.compute_energy_from_K(
            <mycomplex*> &fieldK[0, 0])

    

FFTclass = ${class_name}
