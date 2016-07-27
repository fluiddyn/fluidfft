
include 'base.pyx'


from ${module_name}_pxd cimport (
    ${class_name} as mycppclass,
    mycomplex)


cdef class ${class_name}:
    """Perform fast Fourier transform in 3D.

    """
    cdef mycppclass* thisptr
    cdef tuple _shapeK_loc, _shapeX_loc
    cdef public MPI.Comm comm
    cdef public int nb_proc, rank

    def __cinit__(self, int n0=2, int n1=2, int n2=4):
        self.thisptr = new mycppclass(n0, n1, n2)

    def __init__(self, int n0=2, int n1=2, int n2=4):
        self._shapeK_loc = self.get_shapeK_loc()
        self._shapeX_loc = self.get_shapeX_loc()
        # info on MPI
        self.nb_proc = nb_proc
        self.rank = rank
        if self.nb_proc > 1:
            self.comm = comm

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
            fieldK = np.empty(self._shapeK_loc, dtype=DTYPEc, order='C')
        self.thisptr.fft(&fieldX[0, 0, 0], <mycomplex*> &fieldK[0, 0, 0])
        return np.array(fieldK)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef ifft_generic(self, DTYPEc_t[:, :, ::1] fieldK,
                       DTYPEf_t[:, :, ::1] fieldX=None):
        if fieldX is None:
            fieldX = np.empty(self._shapeX_loc, dtype=DTYPEf, order='C')
        self.thisptr.ifft(<mycomplex*> &fieldK[0, 0, 0], &fieldX[0, 0, 0])
        return np.array(fieldX)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef fft_as_arg(self, DTYPEf_t[:, :, ::1] fieldX,
                     DTYPEc_t[:, :, ::1] fieldK):
        self.thisptr.fft(&fieldX[0, 0, 0], <mycomplex*> &fieldK[0, 0, 0])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef ifft_as_arg(self, DTYPEc_t[:, :, ::1] fieldK,
                      DTYPEf_t[:, :, ::1] fieldX):
        self.thisptr.ifft(<mycomplex*> &fieldK[0, 0, 0], &fieldX[0, 0, 0])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef fft(self, DTYPEf_t[:, :, ::1] fieldX):
        cdef np.ndarray[DTYPEc_t, ndim=3] fieldK
        fieldK = np.empty(self.get_shapeK_loc(), dtype=DTYPEc, order='C')
        self.thisptr.fft(&fieldX[0, 0, 0], <mycomplex*> &fieldK[0, 0, 0])
        return fieldK

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef ifft(self, DTYPEc_t[:, :, ::1] fieldK):
        cdef np.ndarray[DTYPEf_t, ndim=3] fieldX
        fieldX = np.empty(self.get_shapeX_loc(), dtype=DTYPEf, order='C')
        self.thisptr.ifft(<mycomplex*> &fieldK[0, 0, 0], &fieldX[0, 0, 0])
        return fieldX

    cpdef get_shapeX_loc(self):
        """Get the shape of the array in real space for this mpi process."""
        cdef int nX0loc, nX1loc, nX2
        self.thisptr.get_local_shape_X(&nX0loc, &nX1loc, &nX2)
        return nX0loc, nX1loc, nX2

    cpdef get_shapeK_loc(self):
        """Get the shape of the array in Fourier space for this mpi process."""
        cdef int nK0loc, nK1loc, nK2
        self.thisptr.get_local_shape_K(&nK0loc, &nK1loc, &nK2)
        return nK0loc, nK1loc, nK2

    cpdef get_shapeX_seq(self):
        """Get the shape of an array in real space for a sequential run."""
        cdef int nX0, nX1, nX2
        self.thisptr.get_global_shape_X(&nX0, &nX1, &nX2)
        return nX0, nX1, nX2

    def gather_Xspace(self, DTYPEf_t[:, :, ::1] ff_loc,
                      root=None):
        """Gather an array in real space for a parallel run."""
        cdef np.ndarray[DTYPEf_t, ndim=3] ff_seq

        # self.shapeX_loc is the same for all processes,
        # it is safe to use Allgather or Gather
        if root is None:
            ff_seq = np.empty(self.get_shapeX_seq(), DTYPEf)
            self.comm.Allgather(ff_loc, ff_seq)
        elif isinstance(root, int):
            ff_seq = None
            if self.rank == root:
                ff_seq = np.empty(self.get_shapeX_seq(), DTYPEf)
            self.comm.Gather(ff_loc, ff_seq, root=root)
        else:
            raise ValueError('root should be an int')
        return ff_seq

    def scatter_Xspace(self, DTYPEf_t[:, :, ::1] ff_seq,
                      root=None):
        """Gather an array in real space for a parallel run."""
        cdef np.ndarray[DTYPEf_t, ndim=3] ff_loc

        # self.shapeX_loc is the same for all processes,
        # it is safe to use Allgather or Gather
        if root is None:
            ff_loc = np.empty(self.get_shapeX_loc(), DTYPEf)
            self.comm.Scatter(ff_seq, ff_loc, root=0)
        elif isinstance(root, int):
            ff_loc = None
            if self.rank == root:
                ff_loc = np.empty(self.get_shapeX_loc(), DTYPEf)
            self.comm.Scatter(ff_seq, ff_loc, root=root)
        else:
            raise ValueError('root should be an int')
        return ff_loc

    cpdef get_shapeK_seq(self):
        """Get the shape of an array in Fourier space for a sequential run."""
        cdef int nK0, nK1, nK2
        self.thisptr.get_global_shape_K(&nK0, &nK1, &nK2)
        return nK0, nK1, nK2

    def sum_wavenumbers(self, fieldK):
        """Compute the sum over all wavenumbers."""
        if fieldK.dtype == np.float64:
            return self._sum_wavenumbers_double(fieldK)
        elif fieldK.dtype == np.complex128:
            return self._sum_wavenumbers_complex(fieldK)
        else:
            raise TypeError('dtype of fieldK has to be float64 or complex128.')

    cdef _sum_wavenumbers_double(self, DTYPEf_t[:,:,::1] fieldK):
        return self.thisptr.sum_wavenumbers_double(
            <DTYPEf_t*> &fieldK[0, 0, 0])

    cdef _sum_wavenumbers_complex(self, DTYPEc_t[:,:,::1] fieldK):
        cdef DTYPEc_t result
        self.thisptr.sum_wavenumbers_complex(
            <mycomplex*> &fieldK[0, 0, 0], <mycomplex*> &result)
        return result

    cpdef get_dimX_K(self):
        """Get the indices of the real space dimension in Fourier space."""
        cdef int d0, d1, d2
        self.thisptr.get_dimX_K(&d0, &d1, &d2)
        return d0, d1, d2

    cpdef get_seq_indices_first_K(self):
        """Get the "sequential" index of the first number in Fourier space."""
        cdef int i0, i1
        self.thisptr.get_seq_indices_first_K(&i0, &i1)
        return i0, i1

    cpdef get_k_adim_loc(self):
        """Get the non-dimensional wavenumbers stored locally.

        returns k0_adim_loc, k1_adim_loc, k2_adim_loc.

        """
        cdef int nK0, nK1, nK2, nK0_loc, nK1_loc, nK2_loc
        cdef int d0, d1, d2, i0_start, i1_start
        cdef np.ndarray tmp, k0_adim_loc, k1_adim_loc, k2_adim_loc

        nK0, nK1, nK2 = self.get_shapeK_seq()
        nK0_loc, nK1_loc, nK2_loc = self.get_shapeK_loc()

        d0, d1, d2 = self.get_dimX_K()
        i0_start, i1_start = self.get_seq_indices_first_K()

        if d0 == 2:
            tmp = np.arange(nK0)
        else:
            k0_adim_max = nK0//2
            tmp = np.r_[0:k0_adim_max+1, -k0_adim_max+1:0]

        k0_adim_loc = tmp[i0_start:i0_start+nK0_loc]

        if d1 == 2:
            tmp = np.arange(nK1)
        else:
            k1_adim_max = nK1//2
            tmp = np.r_[0:k1_adim_max+1, -k1_adim_max+1:0]

        k1_adim_loc = tmp[i1_start:i1_start+nK1_loc]

        if d2 == 2:
            k2_adim_loc = np.arange(nK2)
        else:
            k2_adim_max = nK2//2
            k2_adim_loc = np.r_[0:k2_adim_max+1, -k2_adim_max+1:0]

        return k0_adim_loc, k1_adim_loc, k2_adim_loc

    def build_invariant_arrayX_from_2d_indices12X(self, o2d, arr2d):

        nX0, nX1, nX2 = self.get_shapeX_seq()
        nX0loc, nX1loc, nX2loc = self.get_shapeX_loc()

        if (nX1, nX2) != tuple(o2d.get_shapeX_seq()):
            raise ValueError('Not the same physical shape...')

        # check that the 2d fft is not with distributed memory...
        if tuple(o2d.get_shapeX_loc()) != tuple(o2d.get_shapeX_loc()):
            raise ValueError('2d fft is with distributed memory...')

        ind0seq_first, ind1seq_first = self.get_seq_indices_first_K()

        if (nX1loc, nX2loc) == o2d.get_shapeX_loc():
            arr3d_loc_2dslice = arr2d
        else:
            raise NotImplementedError

        arr3d = np.empty([nX0loc, nX1loc, nX2loc])
        for i0 in range(nX0loc):
            arr3d[i0] = arr3d_loc_2dslice

        return arr3d

    def build_invariant_arrayK_from_2d_indices12X(self, o2d, arr2d):

        nK0, nK1, nK2 = self.get_shapeK_seq()
        nK0loc, nK1loc, nK2loc = self.get_shapeK_loc()

        nX0, nX1, nX2 = self.get_shapeX_seq()

        if (nX1, nX2) != o2d.get_shapeX_seq():
            raise ValueError('Not the same physical shape...')

        # check that the 2d fft is not with distributed memory...
        if o2d.get_shapeX_loc() != o2d.get_shapeX_loc():
            raise ValueError('2d fft is with distributed memory...')

        ind0seq_first, ind1seq_first = self.get_seq_indices_first_K()
        dimX_K = self.get_dimX_K()

        arr3d = np.zeros([nK0loc, nK1loc, nK2loc], dtype=np.complex128)

        if dimX_K == (0, 1, 2):
            # simple
            if (nK0, nK1, nK2) == (nK0loc, nK1loc, nK2loc):
                # very simple
                arr3d_loc_2dslice = arr2d
            else:
                raise NotImplementedError

            arr3d[0] = arr3d_loc_2dslice

        elif dimX_K == (1, 0, 2):
            # like fft3d.mpi_with_fftwmpi3d
            arr3d_loc_2dslice = np.zeros([nK0loc, nK2loc], dtype=np.complex128)

            for i0 in range(nK0loc):
                for i2 in range(nK2loc):
                    i0_2d = ind0seq_first + i0
                    i1_2d = i2
                    arr3d_loc_2dslice[i0, i2] = arr2d[i0_2d, i1_2d]

            arr3d[:, 0, :] = arr3d_loc_2dslice
        else:
            raise NotImplementedError

        return arr3d


FFTclass = ${class_name}
