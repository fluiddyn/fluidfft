"""Operators 3d (:mod:`fluidfft.fft3d.operators`)
=================================================

.. autoclass:: OperatorsPseudoSpectral3D
   :members:
   :undoc-members:

"""
from __future__ import print_function

from math import pi

from past.builtins import basestring

import numpy as np

from fluiddyn.util import mpi

from fluidfft import create_fft_object, empty_aligned
from fluidfft.util import _rescale_random
from fluidfft.fft2d.operators import _make_str_length

from .util_pythran import (
    project_perpk3d,
    divfft_from_vecfft,
    rotfft_from_vecfft,
    rotfft_from_vecfft_outin,
    vector_product,
    loop_spectra3d,
)

# from .dream_pythran import _vgradv_from_v2

if mpi.nb_proc > 1:
    MPI = mpi.MPI


__all__ = ["vector_product", "OperatorsPseudoSpectral3D"]


class OperatorsPseudoSpectral3D(object):
    """Perform 2D FFT and operations on data.

    Parameters
    ----------

    nx : int

      Global dimension over the x-axis (third dimension for the real arrays).

    ny : int

      Global dimension over the y-axis (second dimension for the real arrays).

    nz : int

      Global dimension over the y-axis (first dimension for the real arrays).

    lx : float

      Length of the domain along the x-axis.

    ly : float

      Length of the domain along the y-axis.

    lz : float

      Length of the domain along the z-axis.

    fft : str or FFT classes

      Name of module or string characterizing a method. It has to correspond to
      a module of fluidfft. The first part "fluidfft." of the module "path" can
      be omitted.

    coef_dealiasing : float

    """

    def __init__(self, nx, ny, nz, lx, ly, lz, fft=None, coef_dealiasing=1.):
        self.nx = self.nx_seq = nx
        self.ny = self.ny_seq = ny
        self.nz = self.nz_seq = nz

        if fft is None:
            if mpi.nb_proc == 1:
                fft = "fft3d.with_pyfftw"
            else:
                fft = "fft3d.mpi_with_fftwmpi3d"

        if isinstance(fft, basestring):
            if fft.lower() in ("sequential", "fftwpy"):
                fft = "fft3d.with_pyfftw"
            if any([fft.startswith(s) for s in ["fluidfft.", "fft3d."]]):
                op_fft = create_fft_object(fft, nz, ny, nx)
            else:
                raise ValueError(
                    "Cannot instantiate {}.".format(fft)
                    + " Expected something like 'fftwpy'"
                    " or 'fluidfft.fft3d.<method>' or 'fft3d.<method>'"
                )

        elif isinstance(fft, type):
            op_fft = fft(nz, ny, nx)
        else:
            op_fft = fft

        self._op_fft = op_fft
        self.type_fft = op_fft.__class__.__module__

        try:
            self.dim_first_fft = op_fft.get_dim_first_fft()
        except AttributeError:
            self.dim_first_fft = 2

        self.shapeX_seq = op_fft.get_shapeX_seq()
        self.shapeX_loc = op_fft.get_shapeX_loc()

        self._is_mpi_lib = self.shapeX_seq != self.shapeX_loc and mpi.nb_proc > 1

        Lx = self.Lx = float(lx)
        Ly = self.Ly = float(ly)
        Lz = self.Lz = float(lz)

        self.deltax = Lx / nx
        self.deltay = Ly / ny
        self.deltaz = Lz / nz

        self.x_seq = self.deltax * np.arange(nx)
        self.y_seq = self.deltay * np.arange(ny)
        self.z_seq = self.deltaz * np.arange(nz)

        self.deltakx = deltakx = 2 * pi / Lx
        self.deltaky = deltaky = 2 * pi / Ly
        self.deltakz = deltakz = 2 * pi / Lz

        self.ifft = self.ifft3d = op_fft.ifft
        self.fft = self.fft3d = op_fft.fft

        self.ifft_as_arg = op_fft.ifft_as_arg
        self.fft_as_arg = op_fft.fft_as_arg

        # try:
        # faster version which destroy the input
        self.ifft_as_arg_destroy = op_fft.ifft_as_arg_destroy
        # except AttributeError:
        #     self.ifft_as_arg_destroy = self.ifft_as_arg

        self.sum_wavenumbers = op_fft.sum_wavenumbers
        self.compute_energy_from_X = op_fft.compute_energy_from_X
        self.compute_energy_from_K = op_fft.compute_energy_from_K

        self.shapeK = self.shapeK_loc = op_fft.get_shapeK_loc()
        self.shapeK_seq = op_fft.get_shapeK_seq()
        self.nk0, self.nk1, self.nk2 = self.shapeK_loc

        order = op_fft.get_dimX_K()
        if order == (0, 1, 2):
            self.deltaks = deltakz, deltaky, deltakx
        elif order == (1, 0, 2):
            self.deltaks = deltaky, deltakz, deltakx
        elif order == (2, 1, 0):
            self.deltaks = deltakx, deltaky, deltakz
        elif order == (1, 2, 0):
            self.deltaks = deltaky, deltakx, deltakz
        else:
            print("order =", order)
            raise NotImplementedError

        for self.dimK_first_fft in range(3):
            if order[self.dimK_first_fft] == self.dim_first_fft:
                break

        k0_adim_loc, k1_adim_loc, k2_adim_loc = op_fft.get_k_adim_loc()

        self.k0 = self.deltaks[0] * k0_adim_loc
        self.k1 = self.deltaks[1] * k1_adim_loc
        self.k2 = self.deltaks[2] * k2_adim_loc

        # oh that's strange!
        K1, K0, K2 = np.meshgrid(self.k1, self.k0, self.k2, copy=False)

        K0 = np.ascontiguousarray(K0)
        K1 = np.ascontiguousarray(K1)
        K2 = np.ascontiguousarray(K2)

        assert K0.shape == self.shapeK_loc, (K0.shape, self.shapeK_loc)

        if order == (0, 1, 2):
            self.Kz = K0
            self.Ky = K1
            self.Kx = K2
        elif order == (1, 0, 2):
            self.Ky = K0
            self.Kz = K1
            self.Kx = K2
        elif order == (2, 1, 0):
            self.Kx = K0
            self.Ky = K1
            self.Kz = K2
        elif order == (1, 2, 0):
            self.Ky = K0
            self.Kx = K1
            self.Kz = K2
        else:
            print("order =", order)
            raise NotImplementedError

        self.K2 = K0 ** 2 + K1 ** 2 + K2 ** 2
        self.K8 = self.K2 ** 4

        self.seq_indices_first_K = op_fft.get_seq_indices_first_K()
        self.seq_indices_first_X = op_fft.get_seq_indices_first_X()

        K_square_nozero = self.K2.copy()

        if all(index == 0 for index in self.seq_indices_first_K):
            K_square_nozero[0, 0, 0] = 1e-14

        self.inv_K_square_nozero = 1. / K_square_nozero

        self.coef_dealiasing = coef_dealiasing

        CONDKX = abs(self.Kx) >= self.coef_dealiasing * self.k2.max()
        CONDKY = abs(self.Ky) >= self.coef_dealiasing * self.k1.max()
        CONDKZ = abs(self.Kz) >= self.coef_dealiasing * self.k0.max()
        where_dealiased = np.logical_or(CONDKX, CONDKY, CONDKZ)
        self.where_dealiased = np.array(where_dealiased, dtype=np.uint8)

        self.gather_Xspace = op_fft.gather_Xspace
        self.scatter_Xspace = op_fft.scatter_Xspace

        if mpi.nb_proc > 1:
            self.comm = mpi.comm

        self.rank = mpi.rank

        # initialization spectra
        self.nkx_spectra = nx // 2 + 1
        self.nky_spectra = ny // 2 + 1
        self.nkz_spectra = nz // 2 + 1

        self.kxmax_spectra = self.deltakx * self.nkx_spectra
        self.kymax_spectra = self.deltaky * self.nky_spectra
        self.kzmax_spectra = self.deltakz * self.nkz_spectra

        self.deltak = self.deltak_spectra3d = max(
            self.deltakx, self.deltaky, self.deltakz
        )
        self.kmax_spectra3d = min(
            self.kxmax_spectra, self.kymax_spectra, self.kzmax_spectra
        )
        self.nk_spectra3d = max(
            2, int(self.kmax_spectra3d / self.deltak_spectra3d)
        )
        self.k_spectra3d = self.deltak_spectra3d * np.arange(self.nk_spectra3d)

    # self.tmp_fields_fft = tuple(self.create_arrayK() for n in range(6))

    def produce_str_describing_grid(self):
        """Produce a short string describing the grid."""
        return "{}x{}x{}".format(self.nx_seq, self.ny_seq, self.nz_seq)

    def produce_str_describing_oper(self):
        """Produce a short string describing the operator."""
        str_Lx = _make_str_length(self.Lx)
        str_Ly = _make_str_length(self.Ly)
        str_Lz = _make_str_length(self.Lz)

        return ("{}x{}x{}_V" + str_Lx + "x" + str_Ly + "x" + str_Lz).format(
            self.nx_seq, self.ny_seq, self.nz_seq
        )

    def produce_long_str_describing_oper(self):
        """Produce a string describing the operator."""

        str_Lx = _make_str_length(self.Lx)
        str_Ly = _make_str_length(self.Ly)
        str_Lz = _make_str_length(self.Lz)

        return (
            "type fft: "
            + self.type_fft
            + "\n"
            + "nx = {:6d} ; ny = {:6d} ; nz = {:6d}\n".format(
                self.nx_seq, self.ny_seq, self.nz_seq
            )
            + "Lx = "
            + str_Lx
            + " ; Ly = "
            + str_Ly
            + " ; Lz = "
            + str_Lz
            + "\n"
        )

    def _get_shapeX(self, shape="loc"):
        if shape.lower() == "loc":
            return self.shapeX_loc

        elif shape.lower() == "seq":
            return self.shapeX_seq

        else:
            raise ValueError('shape should be "loc" or "seq"')

    def _get_shapeK(self, shape="loc"):
        if shape.lower() == "loc":
            return self.shapeK_loc

        elif shape.lower() == "seq":
            return self.shapeK_seq

        else:
            raise ValueError('shape should be "loc" or "seq"')

    def create_arrayX(self, value=None, shape="loc"):
        """Return a constant array in real space."""
        shapeX = self._get_shapeX(shape)
        field = empty_aligned(shapeX)
        if value is not None:
            field.fill(value)
        return field

    def create_arrayK(self, value=None, shape="loc"):
        """Return a constant array in real space."""
        shapeK = self._get_shapeK(shape)
        field = empty_aligned(shapeK, dtype=np.complex128)
        if value is not None:
            field.fill(value)
        return field

    def create_arrayX_random(self, shape="loc", min_val=None, max_val=None):
        """Return a random array in real space."""
        shape = self._get_shapeX(shape)
        values = np.random.random(shape)
        return _rescale_random(values, min_val, max_val)

    def create_arrayK_random(self, shape="loc", min_val=None, max_val=None):
        """Return a random array in real space."""
        shape = self._get_shapeK(shape)
        values = (np.random.random(shape) + 1j * np.random.random(shape))
        return _rescale_random(values, min_val, max_val)

    def sum_wavenumbers_versatile(self, field_fft):
        """Compute the sum over all wavenumbers (versatile version).

        This function should return the same result than
        :func:`sum_wavenumbers`.

        It is here mainly to check that the classes are well implemented.

        """
        spectrum3d_loc = self._compute_spectrum3d_loc(field_fft)
        result = spectrum3d_loc.sum()

        if self._is_mpi_lib:
            result = mpi.comm.allreduce(result, op=mpi.MPI.SUM)

        return result

    def _compute_spectrum3d_loc(self, field_fft):
        """"""

        dimK_first_fft = self.dimK_first_fft

        nx_seq = self.shapeX_seq[self.dim_first_fft]
        # nk_seq = self.shapeK_seq[dimK_first_fft]
        nk_loc = self.shapeK_loc[dimK_first_fft]
        ik_start = self.seq_indices_first_K[dimK_first_fft]
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

    def project_perpk3d(self, vx_fft, vy_fft, vz_fft):
        """Project (inplace) a vector perpendicular to the wavevector.

        The resulting vector is divergence-free.

        """
        project_perpk3d(
            vx_fft,
            vy_fft,
            vz_fft,
            self.Kx,
            self.Ky,
            self.Kz,
            self.inv_K_square_nozero,
        )

    def divfft_from_vecfft(self, vx_fft, vy_fft, vz_fft):
        """Return the divergence of a vector in spectral space."""
        # float64[][][]
        Kx = self.Kx
        Ky = self.Ky
        Kz = self.Kz

        return divfft_from_vecfft(vx_fft, vy_fft, vz_fft, Kx, Ky, Kz)

    def rotfft_from_vecfft(self, vx_fft, vy_fft, vz_fft):
        """Return the curl of a vector in spectral space."""
        # float64[][][]
        Kx = self.Kx
        Ky = self.Ky
        Kz = self.Kz

        return rotfft_from_vecfft(vx_fft, vy_fft, vz_fft, Kx, Ky, Kz)

    def rotfft_from_vecfft_outin(
        self, vx_fft, vy_fft, vz_fft, rotxfft, rotyfft, rotzfft
    ):
        """Return the curl of a vector in spectral space."""
        # float64[][][]
        Kx = self.Kx
        Ky = self.Ky
        Kz = self.Kz

        rotfft_from_vecfft_outin(
            vx_fft, vy_fft, vz_fft, Kx, Ky, Kz, rotxfft, rotyfft, rotzfft
        )

    def div_vb_fft_from_vb(self, vx, vy, vz, b):
        r"""Compute :math:`\nabla \cdot (\boldsymbol{v} b)` in spectral space.

        """
        fft3d = self.fft3d

        vxbfft = fft3d(vx * b)
        vybfft = fft3d(vy * b)
        vzbfft = fft3d(vz * b)

        return divfft_from_vecfft(
            vxbfft, vybfft, vzbfft, self.Kx, self.Ky, self.Kz
        )

    def rotzfft_from_vxvyfft(self, vx_fft, vy_fft):
        """Compute the z component of the curl in spectral space."""
        return 1j * (self.Kx * vy_fft - self.Ky * vx_fft)

    def get_XYZ_loc(self):
        """Compute the local 3d arrays with the x, y, and y values.

        """

        if self.shapeX_seq != self.shapeX_loc:
            i0_seq_start, i1_seq_start, i2_seq_start = self.seq_indices_first_X
            if self.shapeX_seq[1:] != self.shapeX_loc[1:]:
                # general solution
                # mpi.print_sorted(
                #     'in get_XYZ_loc:',
                #     '(i0_seq_start, i1_seq_start, i2_seq_start):',
                #     (i0_seq_start, i1_seq_start, i2_seq_start))

                z_loc = self.z_seq[i0_seq_start:i0_seq_start + self.shapeX_loc[0]]
                y_loc = self.y_seq[i1_seq_start:i1_seq_start + self.shapeX_loc[1]]
                x_loc = self.x_seq[i2_seq_start:i2_seq_start + self.shapeX_loc[2]]

            # mpi.print_sorted('z_loc', z_loc)
            # mpi.print_sorted('y_loc', y_loc)
            # mpi.print_sorted('x_loc', x_loc)

            else:
                # 1d decomposition
                x_loc = self.x_seq
                y_loc = self.y_seq
                z_loc = self.z_seq[i0_seq_start:i0_seq_start + self.shapeX_loc[0]]
        else:
            x_loc = self.x_seq
            y_loc = self.y_seq
            z_loc = self.z_seq

        Y, Z, X = np.meshgrid(y_loc, z_loc, x_loc, copy=False)

        assert X.shape == Y.shape == Z.shape == self.shapeX_loc

        return X, Y, Z

    def compute_1dspectra(self, energy_fft):
        """Compute the 1D spectra.

        Returns
        -------

        spectrum_kx

        spectrum_ky

        spectrum_kz

        """
        # nk0, nk1, nk2 = self.shapeK_loc
        spectrum_k0k1k2 = self._compute_spectrum3d_loc(energy_fft)
        dimX_K = self._op_fft.get_dimX_K()

        if self._is_mpi_lib:

            def compute_spectrum_ki(dimXi):
                ni = self.shapeX_seq[dimXi]
                nk_spectra = ni // 2 + 1
                dimK = dimX_K.index(dimXi)
                dims_for_sum = tuple(dim for dim in range(3) if dim != dimK)
                spectrum_tmp_loc = spectrum_k0k1k2.sum(axis=dims_for_sum)
                istart = self.seq_indices_first_K[dimK]
                nk_loc = self.shapeK_loc[dimK]

                if self.dimK_first_fft != dimK:
                    spectrum_tmp_seq = np.zeros(ni)
                    spectrum_tmp_seq[istart:istart + nk_loc] = spectrum_tmp_loc
                    spectrum_ki = spectrum_tmp_seq[:nk_spectra]
                    nk1 = (ni + 1) // 2
                    spectrum_ki[1:nk1] += spectrum_tmp_seq[nk_spectra:][::-1]
                else:
                    spectrum_tmp_seq = np.zeros(nk_spectra)
                    spectrum_tmp_seq[istart:istart + nk_loc] = spectrum_tmp_loc
                    spectrum_ki = spectrum_tmp_seq

                spectrum_ki = mpi.comm.allreduce(spectrum_ki, op=mpi.MPI.SUM)

                return spectrum_ki

            spectrum_kx = compute_spectrum_ki(dimXi=2)
            spectrum_ky = compute_spectrum_ki(dimXi=1)
            spectrum_kz = compute_spectrum_ki(dimXi=0)

        else:

            def compute_spectrum_ki(dimXi):
                ni = self.shapeX_seq[dimXi]
                nk_spectra = ni // 2 + 1
                dimK = dimX_K.index(dimXi)
                dims_for_sum = tuple(dim for dim in range(3) if dim != dimK)
                spectrum_tmp = spectrum_k0k1k2.sum(axis=dims_for_sum)
                if self.dimK_first_fft != dimK:
                    spectrum_ki = spectrum_tmp[:nk_spectra]
                    nk1 = (ni + 1) // 2
                    spectrum_ki[1:nk1] += spectrum_tmp[nk_spectra:][::-1]
                else:
                    spectrum_ki = spectrum_tmp
                return spectrum_ki

            spectrum_kx = compute_spectrum_ki(dimXi=2)
            spectrum_ky = compute_spectrum_ki(dimXi=1)
            spectrum_kz = compute_spectrum_ki(dimXi=0)

        return (
            spectrum_kx / self.deltakx,
            spectrum_ky / self.deltaky,
            spectrum_kz / self.deltakz,
        )

    def compute_3dspectrum(self, energy_fft):
        """Compute the 3D spectrum.

        The corresponding wavenumber array is ``self.k_spectra3d``.

        """
        K2 = self.K2
        ks = self.k_spectra3d
        spectrum_k0k1k2 = self._compute_spectrum3d_loc(energy_fft)
        spectrum3d = loop_spectra3d(spectrum_k0k1k2, ks, K2)
        if self._is_mpi_lib:
            spectrum3d = mpi.comm.allreduce(spectrum3d, op=mpi.MPI.SUM)
        return spectrum3d / self.deltak_spectra3d

    def compute_spectra_2vars(self, energy_fft):
        """Compute spectra vs 2 variables.

        .. warning::

           Not implemented!

        .. todo::

           Implement the method :func:`compute_spectra_2vars`.

        Returns
        -------

        E_kx_kyz

        E_ky_kzx

        E_kz_kxy

        """
        raise NotImplementedError


# This one is actually not so useful!
# def get_cross_section(self, equation='x=0', to_process=0):
#     """Get a 2d cross section.

#     .. warning::

#        Not implemented!

#     .. todo::

#        Implement the method :func:`get_cross_section`.  We need a
#        not-implemented method :func:`get_seq_indices_first_X` in the C++
#        classes...

#        We first have to implement the very simple cases for which
#        ``equation`` is equal to:

#        - x = 2.
#        - y = 2.
#        - z = 2.
#        - ix = 10
#        - iy = 10
#        - iz = 10

#     Parameters
#     ----------

#     equation: str

#       Equation defining the cross-section. We should be able to use the
#       variables x, y, z, ix, iy and iz.

#     """
#     raise NotImplementedError
