import numpy as np

from fluiddyn.calcul.easypyfft import BaseFFT
from fluiddyn.util import mpi


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
        k_adim_max = nk // 2
        k_adim_min = -((nk - 1) // 2)
        return np.r_[0 : k_adim_max + 1, k_adim_min:0]


class BaseFFTMPI(BaseFFT):
    _dim_first_fft = 2
    _dtype_spatial = np.float64
    _dtype_spectral = np.complex128

    def get_dimX_K(self):
        """Get the indices of the real space dimension in Fourier space."""
        return NotImplemented

    def get_seq_indices_first_K(self):
        """Get the "sequential" indices of the first number in Fourier space."""
        return NotImplemented

    def get_seq_indices_first_X(self):
        """Get the "sequential" indices of the first number in real space."""
        return NotImplemented

    def create_arrayX(self, value=None):
        """Return a constant array in real space."""
        return NotImplemented

    def create_arrayK(self, value=None):
        """Return a constant array in real space."""
        return NotImplemented

    def fft_as_arg(self, fieldX, fieldK):
        """Perform FFT and put result in second argument."""
        return NotImplemented

    def ifft_as_arg(self, fieldK, fieldX):
        """Perform iFFT and put result in second argument."""
        return NotImplemented

    def ifft_as_arg_destroy(self, fieldK, fieldX):
        """Perform iFFT and put result in second argument."""
        return NotImplemented

    def fft(self, fieldX):
        """Perform FFT and return the result."""
        return NotImplemented

    def ifft(self, fieldK):
        """Perform iFFT and return the result."""
        return NotImplemented

    def __init__(self, n0, n1, n2):
        self._shapeK_loc = self.get_shapeK_loc()
        self._shapeX_loc = self.get_shapeX_loc()
        self._shapeK_seq = self.get_shapeK_seq()
        self._shapeX_seq = self.get_shapeX_seq()

        # info on MPI
        self.nb_proc = mpi.nb_proc
        self.rank = mpi.rank
        if mpi.nb_proc > 1:
            self.comm = mpi.comm

        self._is_mpi_lib = self._shapeX_seq != self._shapeX_loc

        order = self.get_dimX_K()
        dim_first_fft = self.get_dim_first_fft()
        for self.dimK_first_fft in range(3):
            if order[self.dimK_first_fft] == dim_first_fft:
                break

    def get_local_size_X(self):
        """Get the local size in real space."""
        shape = self.get_shapeX_loc()
        return shape[0] * shape[1] * shape[2]

    def get_dim_first_fft(self):
        """The dimension (real space) over which the first fft is taken.

        It is usually 2 but it seems to be 0 for p3dfft (written in Fortran!).
        """
        return self._dim_first_fft

    def get_k_adim_loc(self):
        """Get the non-dimensional wavenumbers stored locally.

        Returns
        -------

        k0_adim_loc : np.ndarray

        k1_adim_loc : np.ndarray

        k2_adim_loc :  np.ndarray

        The indices correspond to the index of the dimension in spectral space.

        """

        nK0, nK1, nK2 = self._shapeK_seq
        nK0_loc, nK1_loc, nK2_loc = self._shapeK_loc

        d0, d1, d2 = self.get_dimX_K()
        i0_start, i1_start, i2_start = self.get_seq_indices_first_K()

        k0_adim = compute_k_adim_seq(nK0, d0, self._dim_first_fft)
        k0_adim_loc = k0_adim[i0_start : i0_start + nK0_loc]

        k1_adim = compute_k_adim_seq(nK1, d1, self._dim_first_fft)
        k1_adim_loc = k1_adim[i1_start : i1_start + nK1_loc]

        k2_adim = compute_k_adim_seq(nK2, d2, self._dim_first_fft)
        k2_adim_loc = k2_adim[i2_start : i2_start + nK2_loc]

        return k0_adim_loc, k1_adim_loc, k2_adim_loc

    def build_invariant_arrayX_from_2d_indices12X(self, o2d, arr2d):
        """Build an array in real space invariant in the third dim."""
        nX0, nX1, nX2 = self._shapeX_seq
        nX0loc, nX1loc, nX2loc = self.get_shapeX_loc()

        if (nX1, nX2) != tuple(o2d.shapeX_seq):
            raise ValueError("Not the same physical shape...")

        # check that the 2d fft is not with distributed memory...
        if tuple(o2d.shapeX_seq) != tuple(o2d.shapeX_loc):
            raise ValueError("2d fft is with distributed memory...")

        (
            ind0seq_first,
            ind1seq_first,
            ind2seq_first,
        ) = self.get_seq_indices_first_K()

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
            raise ValueError("Not the same physical shape...")

        # check that the 2d fft is not with distributed memory...
        if o2d.shapeX_seq != o2d.shapeX_loc:
            raise ValueError("2d fft is with distributed memory...")

        (
            ind0seq_first,
            ind1seq_first,
            ind2seq_first,
        ) = self.get_seq_indices_first_K()
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

    def gather_Xspace(self, ff_loc, root=0):
        """Gather an array in real space for a parallel run."""

        if not self._is_mpi_lib:
            return ff_loc

        comm = self.comm

        nX0_loc, nX1_loc, nX2_loc = self.get_shapeX_loc()
        i0_start, i1_start, i2_start = self.get_seq_indices_first_X()

        if root is None:
            ff_seq = np.empty(self._shapeX_seq, self._dtype_spatial)
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
                ff_tmp = np.empty(
                    [nX0_rank, nX1_rank, nX2_loc], self._dtype_spatial
                )
                if self.rank == i:
                    ff_tmp = np.copy(ff_loc)
                ff_tmp = comm.bcast(ff_tmp, root=i)
                ff_seq[
                    i0_startrank : i0_startrank + nX0_rank,
                    i1_startrank : i1_startrank + nX1_rank,
                    :,
                ] = ff_tmp
        elif isinstance(root, int):
            ff_seq = None
            if self.rank == root:
                ff_seq = np.empty(self._shapeX_seq, self._dtype_spatial)
            for i in range(self.nb_proc):
                if i == root and self.rank == root:
                    ff_seq[
                        i0_start : i0_start + nX0_loc,
                        i1_start : i1_start + nX1_loc,
                        :,
                    ] = ff_loc
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
                    ff_tmp = np.empty(
                        [nX0_rank, nX1_rank, nX2_loc], self._dtype_spatial
                    )
                    ff_tmp = comm.recv(source=i)
                    ff_seq[
                        i0_startrank : i0_startrank + nX0_rank,
                        i1_startrank : i1_startrank + nX1_rank,
                        :,
                    ] = ff_tmp
                comm.barrier()
        else:
            raise ValueError("root should be an int")
        return ff_seq

    def scatter_Xspace(self, ff_seq, root=0):
        """Scatter an array in real space for a parallel run."""

        if not self._is_mpi_lib:
            return ff_seq

        comm = self.comm

        if not isinstance(root, int):
            raise ValueError("root should be an int")

        nX0_loc, nX1_loc, nX2_loc = self.get_shapeX_loc()
        i0_start, i1_start, i2_start = self.get_seq_indices_first_X()

        ff_loc = self.create_arrayX()
        for i in range(self.nb_proc):
            if i == root and self.rank == root:
                ff_loc = ff_seq[
                    i0_start : i0_start + nX0_loc,
                    i1_start : i1_start + nX1_loc,
                    :,
                ]
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
                ff_tmp = np.empty(
                    [nX0_rank, nX1_rank, nX2_loc], self._dtype_spatial
                )
                ff_tmp = ff_seq[
                    i0_startrank : i0_startrank + nX0_rank,
                    i1_startrank : i1_startrank + nX1_rank,
                    :,
                ]
                comm.send(ff_tmp, dest=i)
            comm.barrier()
        return ff_loc

    def sum_wavenumbers(self, field_fft):
        """Compute the sum over all wavenumbers (versatile version)."""
        spectrum3d_loc = self._compute_spectrum3d_loc(field_fft)
        result = spectrum3d_loc.sum()

        if self._is_mpi_lib:
            result = mpi.comm.allreduce(result, op=mpi.MPI.SUM)

        return result

    def _compute_spectrum3d_loc(self, field_fft):
        """"""

        dimK_first_fft = self.dimK_first_fft

        nx_seq = self._shapeX_seq[self._dim_first_fft]
        # nk_seq = self.shapeK_seq[dimK_first_fft]
        nk_loc = self._shapeK_loc[dimK_first_fft]
        ik_start = self.get_seq_indices_first_K()[dimK_first_fft]
        ik_stop = ik_start + nk_loc

        # the copy is important: no *= !
        field_fft = 2 * field_fft

        if ik_start == 0:
            if dimK_first_fft == 2:
                slice0 = np.s_[:, :, 0]
            elif dimK_first_fft == 0:
                slice0 = np.s_[0, :, :]
            elif dimK_first_fft == 1:
                slice0 = np.s_[:, 0, :]
            else:
                raise NotImplementedError

            field_fft[slice0] /= 2
        if ik_stop == nx_seq // 2 + 1 and nx_seq % 2 == 0:
            if dimK_first_fft == 2:
                slice_last = np.s_[:, :, -1]
            elif dimK_first_fft == 0:
                slice_last = np.s_[-1, :, :]
            elif dimK_first_fft == 1:
                slice_last = np.s_[:, -1, :]
            else:
                raise NotImplementedError

            field_fft[slice_last] /= 2

        return field_fft

    def compute_energy_from_spatial(self, fieldX):
        """Compute the mean energy from a real space array."""
        energy = fieldX**2

        if not self._is_mpi_lib:
            return energy.mean() / 2

        energy_sum = energy.sum()
        energy_sum = mpi.comm.allreduce(energy_sum, op=mpi.MPI.SUM)
        return energy_sum / np.prod(self.get_shapeX_seq()) / 2

    def compute_energy_from_Fourier(self, fieldK):
        """Compute the mean energy from a Fourier space array."""
        return self.sum_wavenumbers(abs(fieldK) ** 2) / 2

    compute_energy_from_X = compute_energy_from_spatial
    compute_energy_from_K = compute_energy_from_Fourier

    def print_summary_for_debug(self):
        from fluiddyn.util.mpi import print_sorted as print

        print("get_dim_first_fft()", self.get_dim_first_fft())
        print("dimK_first_fft", self.dimK_first_fft)

        print("get_k_adim_loc()", self.get_k_adim_loc())

        print("get_local_size_X()", self.get_local_size_X())
        print("get_seq_indices_first_K()", self.get_seq_indices_first_K())
        print("get_seq_indices_first_X()", self.get_seq_indices_first_X())

        print("get_shapeX_loc()", self.get_shapeX_loc())
        print("get_shapeK_loc()", self.get_shapeK_loc())
        print("get_dimX_K():", self.get_dimX_K())
