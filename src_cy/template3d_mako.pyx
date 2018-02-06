
include 'base.pyx'


from ${module_name} cimport (
    ${class_name} as mycppclass,
    mycomplex)


def compute_k_adim_seq(nk, d):
    if d == 2:
        return np.arange(nk)
    else:
        k_adim_max = nk//2
        k_adim_min = -((nk-1)//2)
        return np.r_[0:k_adim_max+1, k_adim_min:0]


cdef class ${class_name}:
    """Perform fast Fourier transform in 3D.

    """
    cdef int _has_to_destroy
    cdef mycppclass* thisptr
    cdef tuple _shapeK_loc, _shapeX_loc

    IF MPI4PY:
        cdef public MPI.Comm comm
    cdef public int nb_proc, rank

    def __cinit__(self, int n0=2, int n1=2, int n2=4):
        self._has_to_destroy = 1
        try:
            self.thisptr = new mycppclass(n0, n1, n2)
        except ValueError:
            self._has_to_destroy = 0
            raise

    def __init__(self, int n0=2, int n1=2, int n2=4):
        self._shapeK_loc = self.get_shapeK_loc()
        self._shapeX_loc = self.get_shapeX_loc()
        # info on MPI
        self.nb_proc = nb_proc
        self.rank = rank
        if self.nb_proc > 1:
            self.comm = comm

    def __dealloc__(self):
        if self._has_to_destroy:
            self.thisptr.destroy()
        del self.thisptr

    def get_short_name(self):
        """Get a short name of the class"""
        return self.__class__.__name__.lower()

    def get_local_size_X(self):
        """Get the local size in real space"""
        return self.thisptr.get_local_size_X()

    def run_tests(self):
        """Run simple tests from C++"""
        return self.thisptr.test()

    def run_benchs(self, nb_time_execute=10):
        """Run the C++ benchmarcks"""
        cdef DTYPEf_t[:] arr = np.empty([2], DTYPEf)
        self.thisptr.bench(nb_time_execute, &arr[0])
        if rank == 0:
            return arr

    @cython.boundscheck(False)
    @cython.wraparound(False)
    # @cython.initializedcheck(False)
    cpdef fft_as_arg(self, view3df_t fieldX,
                     view3dc_t fieldK):
        """Perform FFT and put result in second argument"""
        self.thisptr.fft(&fieldX[0, 0, 0], <mycomplex*> &fieldK[0, 0, 0])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    # @cython.initializedcheck(False)
    cpdef ifft_as_arg(self, view3dc_t fieldK,
                      view3df_t fieldX):
        """Perform iFFT and put result in second argument"""
        self.thisptr.ifft(<mycomplex*> &fieldK[0, 0, 0], &fieldX[0, 0, 0])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    # @cython.initializedcheck(False)
    cpdef fft(self, view3df_t fieldX):
        """Perform FFT and return the result"""
        cdef np.ndarray[DTYPEc_t, ndim=3] fieldK
        fieldK = np.empty(self.get_shapeK_loc(), dtype=DTYPEc, order='C')
        self.thisptr.fft(&fieldX[0, 0, 0], <mycomplex*> &fieldK[0, 0, 0])
        return fieldK

    @cython.boundscheck(False)
    @cython.wraparound(False)
    # @cython.initializedcheck(False)
    cpdef ifft(self, view3dc_t fieldK):
        """Perform iFFT and return the result"""
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

    def gather_Xspace(self, view3df_t ff_loc, root=None):
        """Gather an array in real space for a parallel run."""
        cdef np.ndarray[DTYPEf_t, ndim=3] ff_seq

        if root is None:
            ff_seq = np.empty(self.get_shapeX_seq(), DTYPEf)
            comm.Allgather(ff_loc, ff_seq)
        elif isinstance(root, int):
            ff_seq = None
            if self.rank == root:
                ff_seq = np.empty(self.get_shapeX_seq(), DTYPEf)
            comm.Gather(ff_loc, ff_seq, root=root)
        else:
            raise ValueError('root should be an int')
        return ff_seq

    def scatter_Xspace(self, view3df_t ff_seq,
                      root=None):
        """Scatter an array in real space for a parallel run."""
        cdef np.ndarray[DTYPEf_t, ndim=3] ff_loc

        if root is None:
            ff_loc = np.empty(self.get_shapeX_loc(), DTYPEf)
            comm.Scatter(ff_seq, ff_loc, root=0)
        elif isinstance(root, int):
            ff_loc = None
            if self.rank == root:
                ff_loc = np.empty(self.get_shapeX_loc(), DTYPEf)
            comm.Scatter(ff_seq, ff_loc, root=root)
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

        k0_adim = compute_k_adim_seq(nK0, d0)
        k0_adim_loc = k0_adim[i0_start:i0_start+nK0_loc]

        k1_adim = compute_k_adim_seq(nK1, d1)
        k1_adim_loc = k1_adim[i1_start:i1_start+nK1_loc]

        k2_adim_loc = compute_k_adim_seq(nK2, d2)

        return k0_adim_loc, k1_adim_loc, k2_adim_loc

    def build_invariant_arrayX_from_2d_indices12X(self, o2d, arr2d):
        """Build an array in real space invariant in the third dim"""
        nX0, nX1, nX2 = self.get_shapeX_seq()
        nX0loc, nX1loc, nX2loc = self.get_shapeX_loc()

        if (nX1, nX2) != tuple(o2d.shapeX_seq):
            raise ValueError('Not the same physical shape...')

        # check that the 2d fft is not with distributed memory...
        if tuple(o2d.shapeX_seq) != tuple(o2d.shapeX_loc):
            raise ValueError('2d fft is with distributed memory...')

        ind0seq_first, ind1seq_first = self.get_seq_indices_first_K()

        if (nX1loc, nX2loc) == o2d.shapeX_loc:
            arr3d_loc_2dslice = arr2d
        else:
            raise NotImplementedError

        arr3d = np.empty([nX0loc, nX1loc, nX2loc])
        for i0 in range(nX0loc):
            arr3d[i0] = arr3d_loc_2dslice

        return arr3d

    def build_invariant_arrayK_from_2d_indices12X(self, o2d, arr2d):
        """Build an array in Fourier space invariant in the third dim"""
        nK0, nK1, nK2 = self.get_shapeK_seq()
        nK0loc, nK1loc, nK2loc = self.get_shapeK_loc()

        nX0, nX1, nX2 = self.get_shapeX_seq()

        if (nX1, nX2) != o2d.shapeX_seq:
            raise ValueError('Not the same physical shape...')

        # check that the 2d fft is not with distributed memory...
        if o2d.shapeX_seq != o2d.shapeX_loc:
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

    def compute_energy_from_X(self, view3df_t fieldX):
        return <float> self.thisptr.compute_energy_from_X(&fieldX[0, 0, 0])

    def compute_energy_from_K(self, view3dc_t fieldK):
        return <float> self.thisptr.compute_energy_from_K(
            <mycomplex*> &fieldK[0, 0, 0])


FFTclass = ${class_name}
