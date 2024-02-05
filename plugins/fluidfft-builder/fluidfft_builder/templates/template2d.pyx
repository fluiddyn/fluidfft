# cython: embedsignature=True
# cython: language_level=3

include 'base.pyx'
${include_base_mpi_pyx}

from ${cpp_name} cimport (
    ${class_name} as mycppclass,
    mycomplex)


cdef class ${class_name}:
    """Perform Fast Fourier Transform in 2d.

    Parameters
    ----------

    n0 : int

      Global size over the first dimension in spatial space. This corresponds
      to the y direction.

    n1 : int

      Global size over the second dimension in spatial space. This corresponds
      to the x direction.

    """
    cdef int _has_to_destroy
    cdef mycppclass* thisptr
    cdef tuple _shapeK_loc, _shapeX_loc, _shapeK_seq, _shapeX_seq
    cdef int _is_mpi_lib

    ${cdef_public_mpi_comm}
    cdef public int nb_proc, rank

    def __cinit__(self, int n0=2, int n1=2):
        self._has_to_destroy = 1
        try:
            self.thisptr = new mycppclass(n0, n1)
        except ValueError:
            self._has_to_destroy = 0
            raise

    def __init__(self, int n0=2, int n1=2):
        self._shapeK_loc = self.get_shapeK_loc()
        self._shapeX_loc = self.get_shapeX_loc()
        self._shapeK_seq = self.get_shapeK_seq()
        self._shapeX_seq = self.get_shapeX_seq()

        # info on MPI
        self.nb_proc = nb_proc
        self.rank = rank
        if nb_proc > 1 and hasattr(self, "comm"):
            self.comm = comm

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
        """Produce a short name of this object."""
        return self.__class__.__name__.lower()

    def get_local_size_X(self):
        """Get the local size in the real space."""
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
    # @cython.initializedcheck(False)
    cpdef fft_as_arg(self, const view2df_t fieldX,
                     view2dc_t fieldK):
        """Perform the fft and copy the result in the second argument."""
        self.thisptr.fft(&fieldX[0, 0], <mycomplex*> &fieldK[0, 0])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    # @cython.initializedcheck(False)
    cpdef ifft_as_arg(self, const view2dc_t fieldK,
                      view2df_t fieldX):
        """Perform the ifft and copy the result in the second argument."""
        self.thisptr.ifft(<mycomplex*> &fieldK[0, 0], &fieldX[0, 0])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    # @cython.initializedcheck(False)
    cpdef fft(self, const view2df_t fieldX):
        """Perform the fft and returns the result."""
        cdef np.ndarray[DTYPEc_t, ndim=2] fieldK
        fieldK = np.empty(self._shapeK_loc, dtype=DTYPEc, order='C')
        self.thisptr.fft(&fieldX[0, 0], <mycomplex*> &fieldK[0, 0])
        return fieldK

    @cython.boundscheck(False)
    @cython.wraparound(False)
    # @cython.initializedcheck(False)
    cpdef ifft(self, const view2dc_t fieldK):
        """Perform the ifft and returns the result."""
        cdef np.ndarray[DTYPEf_t, ndim=2] fieldX
        fieldX = np.empty(self._shapeX_loc, dtype=DTYPEf, order='C')
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
        """Get the "sequential" index of the first number in real space."""
        return <int> self.thisptr.get_local_X0_start(), 0

    def get_seq_indices_first_K(self):
        """Get the "sequential" index of the first number in Fourier space."""
        return <int> self.thisptr.get_local_K0_start(), 0

    def get_k_adim_loc(self):
        """Get the non-dimensional wavenumbers stored locally.

        Returns
        -------

        k0_adim_loc : np.ndarray

        k1_adim_loc : np.ndarray

        The indices correspond to the index of the dimension in spectral space.
        """

        nyseq, nxseq = self._shapeX_seq

        kyseq = np.array(list(range(nyseq//2 + 1)) +
                         list(range(-nyseq//2 + 1, 0)))
        kxseq = np.array(range(nxseq//2 + 1))

        if self.get_is_transposed():
            k0seq, k1seq = kxseq, kyseq
        else:
            k0seq, k1seq = kyseq, kxseq

        ik0_start, ik1_start = self.get_seq_indices_first_K()
        nk0loc, nk1loc = self._shapeK_loc

        k0_adim_loc = k0seq[ik0_start:ik0_start+nk0loc]
        k1_adim_loc = k1seq[ik1_start:ik1_start+nk1loc]

        return k0_adim_loc, k1_adim_loc

    def get_x_adim_loc(self):
        """Get the coordinates of the points stored locally.

        Returns
        -------

        x0loc : np.ndarray

        x1loc : np.ndarray

        The indices correspond to the index of the dimension in real space.
        """
        nyseq, nxseq = self._shapeX_seq

        ix0_start, ix1_start = self.get_seq_indices_first_X()
        nx0loc, nx1loc = self._shapeX_loc

        x0loc = np.array(range(ix0_start, ix0_start+nx0loc))
        x1loc = np.array(range(ix1_start, ix1_start+nx1loc))

        return x0loc, x1loc

    def compute_energy_from_X(self, const view2df_t fieldX):
        """Compute the mean energy from a real space array."""
        return <float> self.thisptr.compute_energy_from_X(&fieldX[0, 0])

    def compute_energy_from_K(self, const view2dc_t fieldK):
        """Compute the mean energy from a Fourier space array."""
        return <float> self.thisptr.compute_energy_from_K(
            <mycomplex*> &fieldK[0, 0])

    def sum_wavenumbers(self, const view2df_t fieldK):
        """Compute the sum over all wavenumbers."""
        return <float> self.thisptr.sum_wavenumbers(&fieldK[0, 0])

    def gather_Xspace(self, ff_loc, root=None):
        """Gather an array in real space for a parallel run."""
        cdef np.ndarray[DTYPEf_t, ndim=2] ff_seq

        if not self._is_mpi_lib:
            return ff_loc

        if ff_loc.shape != self._shapeX_loc:
            raise ValueError(
                "The shape of the local array given is incorrect."
            )

        if root is None:
            ff_seq = np.empty(self._shapeX_seq, DTYPEf)
            self.comm.Allgather(ff_loc, ff_seq)
        elif isinstance(root, int):
            ff_seq = None
            if self.rank == root:
                ff_seq = np.empty(self._shapeX_seq, DTYPEf)
            self.comm.Gather(ff_loc, ff_seq, root=root)
        else:
            raise ValueError('root should be an int')
        return ff_seq

    def scatter_Xspace(self, ff_seq, root=None):
        """Scatter an array in real space for a parallel run."""
        cdef np.ndarray[DTYPEf_t, ndim=2] ff_loc

        if not self._is_mpi_lib:
            return ff_seq

        if ff_seq is not None and ff_seq.shape != self._shapeX_seq:
            raise ValueError(
                "The shape of the sequential array given is incorrect."
            )

        if root is None:
            ff_loc = np.empty(self._shapeX_loc, DTYPEf)
            if self.rank == 0:
                # why do we need that?
                # difference dtype('<f8') and float64?
                ff_seq = ff_seq.astype(DTYPEf)

            self.comm.Scatter(ff_seq, ff_loc, root=0)
        elif isinstance(root, int):
            ff_loc = None
            if self.rank == root:
                ff_loc = np.empty(self._shapeX_loc, DTYPEf)
            self.comm.Scatter(ff_seq, ff_loc, root=root)
        else:
            raise ValueError('root should be an int')
        return ff_loc

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
