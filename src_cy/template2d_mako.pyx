
include 'base.pyx'


from ${module_name}_pxd cimport (
    ${class_name} as mycppclass,
    mycomplex)

from fluiddyn.util import mpi


cdef class ${class_name}:
    """Class to perform Fast Fourier Transform in 2d."""
    cdef int _has_to_destroy
    cdef mycppclass* thisptr
    cdef tuple _shape_K_loc, _shape_X_loc

    def __cinit__(self, int n0=2, int n1=2):
        self._has_to_destroy = 1
        try:
            self.thisptr = new mycppclass(n0, n1)
        except ValueError:
            self._has_to_destroy = 0
            raise
            
    def __init__(self, int n0=2, int n1=2):
        self._shape_K_loc = self.get_shapeK_loc()
        self._shape_X_loc = self.get_shapeX_loc()

    def __dealloc__(self):
        if self._has_to_destroy:
            self.thisptr.destroy()
        del self.thisptr

    def get_short_name(self):
        return self.__class__.__name__.lower()
        
    def get_local_size_X(self):
        return self.thisptr.get_local_size_X()
    
    def run_tests(self):
        """Run the c++ tests."""
        return self.thisptr.test()

    def run_benchs(self, nb_time_execute=10):
        """Run the c++ benchmarcks"""
        cdef DTYPEf_t[:] arr = np.empty([2], DTYPEf)
        self.thisptr.bench(nb_time_execute, &arr[0])
        if rank == 0:
            return arr

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef fft_as_arg(self, DTYPEf_t[:, ::1] fieldX,
                     DTYPEc_t[:, ::1] fieldK):
        """Perform the fft and copy the result in the second argument."""
        self.thisptr.fft(&fieldX[0, 0], <mycomplex*> &fieldK[0, 0])
        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef ifft_as_arg(self, DTYPEc_t[:, ::1] fieldK,
                      DTYPEf_t[:, ::1] fieldX):
        """Perform the ifft and copy the result in the second argument."""
        self.thisptr.ifft(<mycomplex*> &fieldK[0, 0], &fieldX[0, 0])

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
        """Get the local shape of the array in the "real space"."""
        cdef int output_nX0loc, output_nX1
        self.thisptr.get_local_shape_X(&output_nX0loc, &output_nX1)
        return output_nX0loc, output_nX1

    def get_shapeK_loc(self):
        """Get the local shape of the array in the Fourier space"""
        cdef int output_nK0loc, output_nK1
        self.thisptr.get_local_shape_K(&output_nK0loc, &output_nK1)
        return output_nK0loc, output_nK1

    def get_shapeX_seq(self):
        """Get the shape of the real array as it would be with nb_proc = 1"""
        cdef int output_nX0, output_nX1
        self.thisptr.get_shapeX_seq(&output_nX0, &output_nX1)
        return output_nX0, output_nX1

    def get_shapeK_seq(self):
        """Get the shape of the complex array as it would be with nb_proc = 1

        Warning: if get_is_transposed(), the complex array would also be
        transposed, so in this case, one should write:: 

          nKy = self.get_shapeK_seq[1]

        """
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

        kyseq = np.array(list(range(nyseq//2 + 1)) +
                         list(range(-nyseq//2 + 1, 0)))
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

    def sum_wavenumbers(self, DTYPEf_t[:, ::1] fieldK):
        return <float> self.thisptr.sum_wavenumbers(&fieldK[0, 0])

    def gather_Xspace(self, DTYPEf_t[:, ::1] ff_loc, root=None):
        """Gather an array in real space for a parallel run."""
        cdef np.ndarray[DTYPEf_t, ndim=2] ff_seq

        if root is None:
            ff_seq = np.empty(self.get_shapeX_seq(), DTYPEf)
            mpi.comm.Allgather(ff_loc, ff_seq)
        elif isinstance(root, int):
            ff_seq = None
            if self.rank == root:
                ff_seq = np.empty(self.get_shapeX_seq(), DTYPEf)
            mpi.comm.Gather(ff_loc, ff_seq, root=root)
        else:
            raise ValueError('root should be an int')
        return ff_seq

    def scatter_Xspace(self, DTYPEf_t[:, ::1] ff_seq, root=None):
        """Scatter an array in real space for a parallel run."""
        cdef np.ndarray[DTYPEf_t, ndim=2] ff_loc

        if root is None:
            ff_loc = np.empty(self.get_shapeX_loc(), DTYPEf)
            mpi.comm.Scatter(ff_seq, ff_loc, root=0)
        elif isinstance(root, int):
            ff_loc = None
            if self.rank == root:
                ff_loc = np.empty(self.get_shapeX_loc(), DTYPEf)
            mpi.comm.Scatter(ff_seq, ff_loc, root=root)
        else:
            raise ValueError('root should be an int')
        return ff_loc
    

FFTclass = ${class_name}
