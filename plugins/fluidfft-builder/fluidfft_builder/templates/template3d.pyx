# cython: embedsignature=True
# cython: language_level=3

include 'base.pyx'
${include_base_mpi_pyx}

from ${cpp_name} cimport (
    ${class_name} as mycppclass,
    mycomplex)


def compute_k_adim_seq(nk, axis, dim_first_fft=2):
    """Compute the adimensional wavenumber for an axis.

    Parameters
    ----------

    nk : int

      Global size in Fourier space for the axis.

    axis : int

      Index of the axis in real space (0 for z, 1 for y and 2 for x).

    """
    if axis == dim_first_fft:
        return np.arange(nk)
    else:
        k_adim_max = nk//2
        k_adim_min = -((nk-1)//2)
        return np.r_[0:k_adim_max+1, k_adim_min:0]


cdef class ${class_name}:
    """Perform Fast Fourier Transform in 3D.

    Parameters
    ----------

    n0 : int

      Global size over the first dimension in spatial space. This corresponds
      to the z direction.

    n1 : int

      Global size over the second dimension in spatial space. This corresponds
      to the y direction.

    n2 : int

      Global size over the second dimension in spatial space. This corresponds
      to the x direction.

    """
    cdef int _has_to_destroy
    cdef mycppclass* thisptr
    cdef tuple _shapeK_loc, _shapeX_loc, _shapeK_seq, _shapeX_seq
    cdef int dim_first_fft
    cdef int _is_mpi_lib

    ${cdef_public_mpi_comm}

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
        self._shapeK_seq = self.get_shapeK_seq()
        self._shapeX_seq = self.get_shapeX_seq()

        # info on MPI
        self.nb_proc = nb_proc
        self.rank = rank
        if nb_proc > 1:
            self.comm = comm

        self.dim_first_fft = 2
        self._is_mpi_lib = self._shapeX_seq != self._shapeX_loc

    def __dealloc__(self):
        if self._has_to_destroy:
            self.thisptr.destroy()
        del self.thisptr

    @property
    def _numpy_api(self):
        """A ``@property`` which imports and returns a NumPy-like array backend."""
        import ${numpy_api} as np
        return np

    def get_short_name(self):
        """Get a short name of the class."""
        return self.__class__.__name__.lower()

    def get_dim_first_fft(self):
        """The dimension (real space) over which the first fft is taken.

        It is usually 2 but it seems to be 0 for p3dfft (written in Fortran!).
        """
        return self.dim_first_fft

    def get_local_size_X(self):
        """Get the local size in real space."""
        return self.thisptr.get_local_size_X()

    def run_tests(self):
        """Run simple tests from C++."""
        return self.thisptr.test()

    def run_benchs(self, nb_time_execute=10):
        """Run the C++ benchmarcks."""
        cdef DTYPEf_t[:] arr = np.empty([2], DTYPEf)
        self.thisptr.bench(nb_time_execute, &arr[0])
        if rank == 0:
            return arr

    @cython.boundscheck(False)
    @cython.wraparound(False)
    # @cython.initializedcheck(False)
    cpdef fft_as_arg(self, const view3df_t fieldX,
                     view3dc_t fieldK):
        """Perform FFT and put result in second argument."""

        # if not (is_byte_aligned(fieldX) and is_byte_aligned(fieldK)):
        #     raise ValueError('Requires aligned array.')

        self.thisptr.fft(&fieldX[0, 0, 0], <mycomplex*> &fieldK[0, 0, 0])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    # @cython.initializedcheck(False)
    cpdef ifft_as_arg(self, const view3dc_t fieldK,
                      view3df_t fieldX):
        """Perform iFFT and put result in second argument.

        .. note::
            The values in the input array would be retained by making a copy to
            an intermediate input array. This can have a performance impact.

        """
        self.thisptr.ifft(<mycomplex*> &fieldK[0, 0, 0], &fieldX[0, 0, 0])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    # @cython.initializedcheck(False)
    cpdef ifft_as_arg_destroy(self, view3dc_t fieldK,
                              view3df_t fieldX):
        """Perform iFFT and put result in second argument.

        .. note::
            The values in the input array would be destroyed for the better
            performance.

        """
        self.thisptr.ifft_destroy(
            <mycomplex*> &fieldK[0, 0, 0], &fieldX[0, 0, 0])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    # @cython.initializedcheck(False)
    cpdef fft(self, const view3df_t fieldX):
        """Perform FFT and return the result."""
        cdef np.ndarray[DTYPEc_t, ndim=3] fieldK
        fieldK = np.empty(self._shapeK_loc, dtype=DTYPEc, order='C')
        self.thisptr.fft(&fieldX[0, 0, 0], <mycomplex*> &fieldK[0, 0, 0])
        return fieldK

    @cython.boundscheck(False)
    @cython.wraparound(False)
    # @cython.initializedcheck(False)
    cpdef ifft(self, const view3dc_t fieldK):
        """Perform iFFT and return the result."""
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

    def gather_Xspace(self, ff_loc, root=0):
        """Gather an array in real space for a parallel run.
        """
        cdef np.ndarray[DTYPEf_t, ndim=3] ff_seq
        cdef np.ndarray[DTYPEf_t, ndim=3] ff_tmp
        cdef int nX0_loc, nX1_loc, nX2_loc
        cdef int i0_start, i1_start, i2_start, i
        cdef int i0_startrank, i1_startrank, nX0_rank, nX1_rank

        if not self._is_mpi_lib:
            return ff_loc

        nX0_loc, nX1_loc, nX2_loc = self.get_shapeX_loc()
        i0_start, i1_start, i2_start = self.get_seq_indices_first_X()

        if root is None:
            ff_seq = np.empty(self._shapeX_seq, DTYPEf)
            for i in range(self.nb_proc):
                if self.rank == i:
                    nX0_rank = nX0_loc
                    nX1_rank = nX1_loc
                    i0_startrank = i0_start
                    i1_startrank = i1_start
                nX0_rank = comm.bcast(nX0_rank, root=i)
                nX1_rank = comm.bcast(nX1_rank, root=i)
                i0_startrank = comm.bcast(i0_startrank, root=i)
                i1_startrank = comm.bcast(i1_startrank, root=i)
                ff_tmp = np.empty([nX0_rank, nX1_rank, nX2_loc], DTYPEf)
                if self.rank == i:
                    ff_tmp = np.copy(ff_loc)
                ff_tmp = comm.bcast(ff_tmp, root=i)
                ff_seq[i0_startrank:i0_startrank+nX0_rank,
                       i1_startrank:i1_startrank+nX1_rank, :] = ff_tmp
        elif isinstance(root, int):
            ff_seq = None
            if self.rank == root:
                ff_seq = np.empty(self._shapeX_seq, DTYPEf)
            for i in range(self.nb_proc):
                if i == root and self.rank == root:
                    ff_seq[i0_start:i0_start+nX0_loc,
                           i1_start:i1_start+nX1_loc, :] = ff_loc
                elif self.rank == i:
                    nX0_rank = nX0_loc
                    nX1_rank = nX1_loc
                    i0_startrank = i0_start
                    i1_startrank = i1_start
                    comm.send(nX0_rank, dest=root)
                    comm.send(nX1_rank, dest=root)
                    comm.send(i0_startrank, dest=root)
                    comm.send(i1_startrank, dest=root)
                    comm.send(ff_loc, dest=root)
                elif self.rank == root:
                    nX0_rank = comm.recv(source=i)
                    nX1_rank = comm.recv(source=i)
                    i0_startrank = comm.recv(source=i)
                    i1_startrank = comm.recv(source=i)
                    ff_tmp = np.empty([nX0_rank, nX1_rank, nX2_loc], DTYPEf)
                    ff_tmp = comm.recv(source=i)
                    ff_seq[i0_startrank:i0_startrank+nX0_rank,
                           i1_startrank:i1_startrank+nX1_rank, :] = ff_tmp
                comm.barrier()
        else:
            raise ValueError('root should be an int')
        return ff_seq

    def scatter_Xspace(self, ff_seq, root=0):
        """Scatter an array in real space for a parallel run.

        """
        cdef np.ndarray[DTYPEf_t, ndim=3] ff_loc
        cdef int nX0_loc, nX1_loc, nX2_loc
        cdef int i0_start, i1_start, i2_start, i
        cdef int i0_startrank, i1_startrank, nX0_rank, nX1_rank

        if not self._is_mpi_lib:
            return ff_seq

        if not isinstance(root, int):
            raise ValueError('root should be an int')

        nX0_loc, nX1_loc, nX2_loc = self.get_shapeX_loc()
        i0_start, i1_start, i2_start = self.get_seq_indices_first_X()

        ff_loc = np.empty(self.get_shapeX_loc(), DTYPEf)
        for i in range(self.nb_proc):
            if i == root and self.rank == root:
                ff_loc = ff_seq[i0_start:i0_start+nX0_loc,
                                i1_start:i1_start+nX1_loc, :]
            elif self.rank == i:
                nX0_rank = nX0_loc
                nX1_rank = nX1_loc
                i0_startrank = i0_start
                i1_startrank = i1_start
                comm.send(nX0_rank, dest=root)
                comm.send(nX1_rank, dest=root)
                comm.send(i0_startrank, dest=root)
                comm.send(i1_startrank, dest=root)
                ff_loc = comm.recv(source=root)
            elif self.rank == root:
                nX0_rank = comm.recv(source=i)
                nX1_rank = comm.recv(source=i)
                i0_startrank = comm.recv(source=i)
                i1_startrank = comm.recv(source=i)
                ff_tmp = np.empty([nX0_rank, nX1_rank, nX2_loc], DTYPEf)
                ff_tmp = ff_seq[i0_startrank:i0_startrank+nX0_rank,
                                i1_startrank:i1_startrank+nX1_rank, :]
                comm.send(ff_tmp, dest=i)
            comm.barrier()
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
        """Get the "sequential" indices of the first number in Fourier space."""
        cdef int i0, i1, i2
        self.thisptr.get_seq_indices_first_K(&i0, &i1, &i2)
        return i0, i1, i2

    cpdef get_seq_indices_first_X(self):
        """Get the "sequential" indices of the first number in real space."""
        cdef int i0, i1, i2
        self.thisptr.get_seq_indices_first_X(&i0, &i1, &i2)
        return i0, i1, i2

    cpdef get_k_adim_loc(self):
        """Get the non-dimensional wavenumbers stored locally.

        Returns
        -------

        k0_adim_loc : np.ndarray

        k1_adim_loc : np.ndarray

        k2_adim_loc :  np.ndarray

        The indices correspond to the index of the dimension in spectral space.

        """
        cdef int nK0, nK1, nK2, nK0_loc, nK1_loc, nK2_loc
        cdef int d0, d1, d2, i0_start, i1_start, i2_start
        cdef np.ndarray tmp, k0_adim_loc, k1_adim_loc, k2_adim_loc

        nK0, nK1, nK2 = self._shapeK_seq
        nK0_loc, nK1_loc, nK2_loc = self._shapeK_loc

        d0, d1, d2 = self.get_dimX_K()
        i0_start, i1_start, i2_start = self.get_seq_indices_first_K()

        k0_adim = compute_k_adim_seq(nK0, d0, self.dim_first_fft)
        k0_adim_loc = k0_adim[i0_start:i0_start+nK0_loc]

        k1_adim = compute_k_adim_seq(nK1, d1, self.dim_first_fft)
        k1_adim_loc = k1_adim[i1_start:i1_start+nK1_loc]

        k2_adim = compute_k_adim_seq(nK2, d2, self.dim_first_fft)
        k2_adim_loc = k2_adim[i2_start:i2_start+nK2_loc]

        return k0_adim_loc, k1_adim_loc, k2_adim_loc

    def build_invariant_arrayX_from_2d_indices12X(self, o2d, arr2d):
        """Build an array in real space invariant in the third dim."""
        nX0, nX1, nX2 = self._shapeX_seq
        nX0loc, nX1loc, nX2loc = self.get_shapeX_loc()

        if (nX1, nX2) != tuple(o2d.shapeX_seq):
            raise ValueError('Not the same physical shape...')

        # check that the 2d fft is not with distributed memory...
        if tuple(o2d.shapeX_seq) != tuple(o2d.shapeX_loc):
            raise ValueError('2d fft is with distributed memory...')

        ind0seq_first, ind1seq_first, ind2seq_first = \
            self.get_seq_indices_first_K()

        if (nX1loc, nX2loc) == o2d.shapeX_loc:
            arr3d_loc_2dslice = arr2d
        else:
            raise NotImplementedError

        arr3d = np.empty([nX0loc, nX1loc, nX2loc])
        for i0 in range(nX0loc):
            arr3d[i0] = arr3d_loc_2dslice

        return arr3d

    def build_invariant_arrayK_from_2d_indices12X(self, o2d, arr2d):
        """Build an array in Fourier space invariant in the third dim."""
        nK0, nK1, nK2 = self._shapeK_seq
        nK0loc, nK1loc, nK2loc = self._shapeK_loc

        nX0, nX1, nX2 = self._shapeX_seq

        if (nX1, nX2) != o2d.shapeX_seq:
            raise ValueError('Not the same physical shape...')

        # check that the 2d fft is not with distributed memory...
        if o2d.shapeX_seq != o2d.shapeX_loc:
            raise ValueError('2d fft is with distributed memory...')

        ind0seq_first, ind1seq_first, ind2seq_first = \
            self.get_seq_indices_first_K()
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

    def compute_energy_from_X(self, const view3df_t fieldX):
        """Compute the mean energy from a real space array."""
        return <float> self.thisptr.compute_energy_from_X(&fieldX[0, 0, 0])

    def compute_energy_from_K(self, const view3dc_t fieldK):
        """Compute the mean energy from a Fourier space array."""
        return <float> self.thisptr.compute_energy_from_K(
            <mycomplex*> &fieldK[0, 0, 0])

    def create_arrayX(self, value=None, shape=None):
        """Return a constant array in real space."""
        if shape is None:
            shape = self._shapeX_loc

        field = empty_aligned(shape)
        if value is not None:
            field.fill(value)
        return field

    def create_arrayK(self, value=None, shape=None):
        """Return a constant array in real space."""
        if shape is None:
            shape = self._shapeK_loc

        field = empty_aligned(shape, dtype=np.complex128)
        if value is not None:
            field.fill(value)
        return field

FFTclass = ${class_name}
