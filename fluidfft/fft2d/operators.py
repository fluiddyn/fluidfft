"""Operators 2d (:mod:`fluidfft.fft2d.operators`)
=================================================

.. autoclass:: OperatorsPseudoSpectral2D
   :members:
   :undoc-members:

"""
from __future__ import print_function

from builtins import range

from math import pi
import warnings

from past.builtins import basestring

import numpy as np

from fluiddyn.util import mpi

from fluidfft import create_fft_object, empty_aligned
from fluidfft.util import _rescale_random

from .util_pythran import (
    dealiasing_variable,
    vecfft_from_rotfft,
    vecfft_from_divfft,
    gradfft_from_fft,
    divfft_from_vecfft,
    rotfft_from_vecfft,
)

if mpi.nb_proc > 1:
    MPI = mpi.MPI


def _make_str_length(length):
    l_over_pi = length / np.pi
    if l_over_pi.is_integer():
        return repr(int(l_over_pi)) + "pi"

    elif round(length) == length:
        return "{}".format(int(length))

    return "{:.3f}".format(length).rstrip("0")


class OperatorsPseudoSpectral2D(object):
    """Perform 2D FFT and operations on data.

    Parameters
    ----------

    nx : int

      Global dimension over the x-axis (second dimension for the real arrays).

    ny : int

      Global dimension over the y-axis (first dimension for the real arrays).

    lx : float

      Length of the domain along the x-axis.

    ly : float

      Length of the domain along the y-axis.

    fft : str or FFT classes

      Name of module or string characterizing a method. It has to correspond to a
      module of fluidfft. The first part "fluidfft." of the module "path" can be
      omitted.

    coef_dealiasing : float

    """

    def __init__(self, nx, ny, lx, ly, fft=None, coef_dealiasing=1.):

        self.nx_seq = self.nx = nx = int(nx)
        self.ny_seq = self.ny = ny = int(ny)
        self.lx = lx = float(lx)
        self.ly = ly = float(ly)

        if fft is None:
            if mpi.nb_proc == 1:
                fft = "fft2d.with_pyfftw"
            else:
                fft = "fft2d.mpi_with_fftwmpi2d"

        if isinstance(fft, basestring):
            if fft.lower() == "sequential":
                fft = "fft2d.with_pyfftw"

            if any([fft.startswith(s) for s in ["fluidfft.", "fft2d."]]):
                opfft = create_fft_object(fft, ny, nx)
            else:
                raise ValueError(
                    (
                        "Cannot instantiate %s. Expected something like "
                        "'sequential', 'fluidfft.fft2d.<method>' or "
                        "'fft2d.<method>'"
                    )
                    % fft
                )

        elif isinstance(fft, type):
            opfft = fft(ny, nx)
        else:
            opfft = fft

        self.opfft = opfft
        self.type_fft = opfft.__class__.__module__

        self.is_transposed = opfft.get_is_transposed()

        self.fft2 = self.fft = self.opfft.fft
        self.ifft2 = self.ifft = self.opfft.ifft

        self.fft_as_arg = opfft.fft_as_arg
        self.ifft_as_arg = opfft.ifft_as_arg
        self.shapeX = self.shapeX_loc = opfft.get_shapeX_loc()
        self.shapeX_seq = opfft.get_shapeX_seq()
        self.shapeK = self.shapeK_loc = opfft.get_shapeK_loc()
        self.shapeK_seq = opfft.get_shapeK_seq()
        self.compute_energy_from_X = opfft.compute_energy_from_X
        self.compute_energy_from_K = opfft.compute_energy_from_K

        self.spectrum2D_from_fft = self.compute_2dspectrum
        self.spectra1D_from_fft = self.compute_1dspectra

        self.seq_indices_first_K = opfft.get_seq_indices_first_K()
        self.seq_indices_first_X = opfft.get_seq_indices_first_X()

        self.deltax = lx / nx
        self.deltay = ly / ny
        self.x_seq = self.x = self.deltax * np.arange(nx)
        self.y_seq = self.y = self.deltay * np.arange(ny)

        self.deltakx = 2 * pi / lx
        self.deltaky = 2 * pi / ly

        x0_adim, x1_adim = opfft.get_x_adim_loc()
        self.nX0_loc = len(x0_adim)
        self.nX1_loc = len(x1_adim)

        self.x = self.x_loc = self.deltax * x1_adim
        self.y = self.y_loc = self.deltay * x0_adim

        [self.X, self.Y] = np.meshgrid(self.x_loc, self.y_loc)

        k0_adim, k1_adim = opfft.get_k_adim_loc()

        self.nK0_loc = len(k0_adim)
        self.nK1_loc = len(k1_adim)

        if self.nK0_loc == 0 or self.nK1_loc == 0:
            warnings.warn(
                "The shape for processor {} is {}. ".format(
                    mpi.rank, self.shapeK_loc
                )
                + "This means that it has no data to treat. "
                "It is not efficient and many of the functions of "
                "this operators won't work. Unless you know what you do, "
                "change the fft class, the resolution or "
                "the number of MPI processes."
            )

        if self.is_transposed:
            kx_adim = k0_adim
            ky_adim = k1_adim
        else:
            kx_adim = k1_adim
            ky_adim = k0_adim

        self.kx = self.deltakx * kx_adim
        self.ky = self.deltaky * ky_adim

        self.kx_loc = self.kx
        self.ky_loc = self.ky

        if not self.is_transposed:
            [self.KX, self.KY] = np.meshgrid(self.kx_loc, self.ky_loc)
            self.dim_kx = 1
            self.dim_ky = 0
            self.k0 = self.ky
            self.k1 = self.kx
            self.nkx_loc = self.nK1_loc
            self.nky_loc = self.nK0_loc
        else:
            [self.KY, self.KX] = np.meshgrid(self.ky_loc, self.kx_loc)
            self.dim_kx = 0
            self.dim_ky = 1
            self.k0 = self.kx
            self.k1 = self.ky
            self.nkx_loc = self.nK0_loc
            self.nky_loc = self.nK1_loc

        assert self.KX.shape == self.shapeK_loc

        self.KX2 = self.KX ** 2
        self.KY2 = self.KY ** 2
        self.K2 = self.KX2 + self.KY2
        self.K4 = self.K2 ** 2
        self.K8 = self.K4 ** 2
        self.K = np.sqrt(self.K2)

        self.K_not0 = self.K.copy()
        self.K2_not0 = self.K2.copy()
        self.K4_not0 = self.K4.copy()

        self.is_sequential = opfft.get_shapeK_loc() == opfft.get_shapeK_seq()

        self.rank = mpi.rank
        self.nb_proc = mpi.nb_proc

        if mpi.nb_proc > 1:
            self.comm = mpi.comm

        if not self.is_sequential:
            self.gather_Xspace = self.opfft.gather_Xspace
            self.scatter_Xspace = self.opfft.scatter_Xspace

        if mpi.rank == 0 or self.is_sequential:
            self.K_not0[0, 0] = 10.e-10
            self.K2_not0[0, 0] = 10.e-10
            self.K4_not0[0, 0] = 10.e-10

        self.KX_over_K2 = self.KX / self.K2_not0
        self.KY_over_K2 = self.KY / self.K2_not0

        self.nkx_seq = nx // 2 + 1
        self.nky_seq = ny
        self.nky_spectra = ny // 2 + 1

        khmax = min(self.deltakx * self.nkx_seq, self.deltakx * self.nky_spectra)
        self.deltak = max(self.deltakx, self.deltaky)
        self.nkh = int(khmax / self.deltak)
        if self.nkh == 0:
            self.nkh = 1
        self.kh_2dspectrum = self.deltak * np.arange(self.nkh)

        # Initialisation dealiasing
        kx_max = self.deltakx * (nx // 2)
        ky_max = self.deltaky * (ny // 2)
        self.coef_dealiasing = coef_dealiasing

        if isinstance(coef_dealiasing, bool) and not coef_dealiasing:
            self._has_to_dealiase = False
            self.kxmax_dealiasing = kx_max + self.deltakx
            self.kymax_dealiasing = ky_max + self.deltaky
        else:
            self._has_to_dealiase = True
            self.kxmax_dealiasing = coef_dealiasing * kx_max
            self.kymax_dealiasing = coef_dealiasing * ky_max
        CONDKX = abs(self.KX) >= self.kxmax_dealiasing
        CONDKY = abs(self.KY) >= self.kymax_dealiasing
        where_dealiased = np.logical_or(CONDKX, CONDKY)
        self.where_dealiased = np.array(where_dealiased, dtype=np.uint8)
        self.indexes_dealiased = np.argwhere(where_dealiased)

        # for spectra
        self.nkxE = self.nx_seq // 2 + 1
        self.nkxE2 = (self.nx_seq + 1) // 2
        self.nkyE = self.ny_seq // 2 + 1
        self.nkyE2 = (self.ny_seq + 1) // 2

        # print('nkxE, nkxE2', self.nkxE, self.nkxE2)
        # print('nkyE, nkyE2', self.nkyE, self.nkyE2)

        self.kxE = self.deltakx * np.arange(self.nkxE)
        self.kyE = self.deltaky * np.arange(self.nkyE)
        self.khE = self.kxE
        self.nkhE = self.nkxE

        y_loc, x_loc = self.opfft.get_x_adim_loc()
        y_loc = y_loc * self.deltay
        x_loc = x_loc * self.deltax

        self.XX, self.YY = np.meshgrid(x_loc, y_loc)

    def sum_wavenumbers(self, field_fft):
        """Compute the sum over all wavenumbers."""
        return self.opfft.sum_wavenumbers(np.ascontiguousarray(field_fft))

    def sum_wavenumbers_versatile(self, field_fft):
        """Compute the sum over all wavenumbers (versatile version).

        This function should return the same result than
        :func:`sum_wavenumbers`.

        It is here mainly to check that the classes are well implemented.
        """
        raise NotImplementedError

    def produce_str_describing_grid(self):
        """Produce a short string describing the grid."""
        return "{}x{}".format(self.nx_seq, self.ny_seq)

    def produce_str_describing_oper(self):
        """Produce a short string describing the operator."""
        str_lx = _make_str_length(self.lx)
        str_ly = _make_str_length(self.ly)
        return ("{}x{}_S" + str_lx + "x" + str_ly).format(
            self.nx_seq, self.ny_seq
        )

    def produce_long_str_describing_oper(self):
        """Produce a string describing the operator."""
        str_lx = _make_str_length(self.lx)
        str_ly = _make_str_length(self.ly)
        return (
            "type fft: "
            + str(self.type_fft)
            + "\n"
            + "nx = {0:6d} ; ny = {1:6d}\n".format(self.nx_seq, self.ny_seq)
            + "lx = "
            + str_lx
            + " ; ly = "
            + str_ly
            + "\n"
        )

    def compute_1dspectra(self, energy_fft):
        """Compute the 1D spectra. Return a dictionary."""
        if self.is_sequential and not self.is_transposed:
            # Memory is not shared
            # In this case, self.dim_ky == 0 and self.dim_kx == 1
            # note that only the kx >= 0 are in the spectral variables
            #
            # computation of E_kx
            # we sum over all ky
            # the 2 is here because there are only the kx >= 0
            E_kx = 2. * energy_fft.sum(self.dim_ky) / self.deltakx
            E_kx[0] = E_kx[0] / 2
            if self.nx_seq % 2 == 0:
                E_kx[-1] = E_kx[-1] / 2
            E_kx = E_kx[:self.nkxE]
            # computation of E_ky
            E_ky_tmp = energy_fft[:, 0].copy()
            E_ky_tmp += 2 * energy_fft[:, 1:self.nkxE2].sum(1)
            if self.nx_seq % 2 == 0:
                E_ky_tmp += energy_fft[:, self.nkxE - 1]
            nkyE = self.nkyE
            E_ky = E_ky_tmp[:nkyE]
            E_ky[1:self.nkyE2] += E_ky_tmp[self.nkyE:self.nky_seq][::-1]
            E_ky = E_ky / self.deltaky
        elif self.is_sequential and self.is_transposed:
            # Memory is not shared
            # In this case, self.dim_ky == 1 and self.dim_kx == 0
            # note that only the kx >= 0 are in the spectral variables
            #
            # computation of E_kx
            # we sum over all ky
            # the 2 is here because there are only the kx >= 0
            E_kx = 2. * energy_fft.sum(self.dim_ky) / self.deltakx
            E_kx[0] = E_kx[0] / 2
            if self.nx_seq % 2 == 0 and self.shapeK_seq[0] == self.nkxE:
                E_kx[-1] = E_kx[-1] / 2
            E_kx = E_kx[:self.nkxE]
            # computation of E_ky
            E_ky_tmp = energy_fft[0, :].copy()
            E_ky_tmp += 2 * energy_fft[1:self.nkxE2, :].sum(0)
            if self.nx_seq % 2 == 0 and self.shapeK_seq[0] == self.nkxE:
                E_ky_tmp += energy_fft[self.nkxE - 1, :]
            nkyE = self.nkyE
            E_ky = E_ky_tmp[:nkyE]
            E_ky[1:self.nkyE2] += E_ky_tmp[self.nkyE:self.nky_seq][::-1]
            E_ky = E_ky / self.deltaky
        elif self.is_transposed:
            # Memory is shared along kx (dim 0)
            # In this case, self.dim_ky == 1 and self.dim_kx == 0
            # note that only the kx >= 0 are in the spectral variables
            #
            # computation of E_kx
            # we sum over all ky
            # the 2 is here because there are only the kx >= 0
            E_kx_loc = 2. * energy_fft.sum(self.dim_ky) / self.deltakx
            if self.rank == 0:
                E_kx_loc[0] = E_kx_loc[0] / 2

            if self.shapeK_loc[0] != 1:
                if (
                    self.rank == self.nb_proc - 1
                    and self.nx_seq % 2 == 0
                    and self.shapeK_seq[0] == self.nkxE
                ):
                    E_kx_loc[-1] = E_kx_loc[-1] / 2

            E_kx = np.zeros(self.nkxE)
            counts = self.comm.allgather(self.nkx_loc)
            self.comm.Allgatherv(
                sendbuf=[E_kx_loc, MPI.DOUBLE],
                recvbuf=[E_kx, (counts, None), MPI.DOUBLE],
            )
            E_kx = E_kx[:self.nkxE]

            # computation of E_ky
            E_ky_tmp = 2 * energy_fft[1:-1, :].sum(0)

            if self.rank == 0:
                E_ky_tmp += energy_fft[0, :]
            else:
                E_ky_tmp += 2 * energy_fft[0, :]

            if self.shapeK_loc[0] != 1:
                if (
                    self.rank == self.nb_proc - 1
                    and self.nx_seq % 2 == 0
                    and self.shapeK_seq[0] == self.nkxE
                ):
                    E_ky_tmp += energy_fft[-1, :]
                else:
                    E_ky_tmp += 2 * energy_fft[-1, :]

            nkyE = self.nkyE
            E_ky = E_ky_tmp[:nkyE]
            E_ky[1:self.nkyE2] += E_ky_tmp[self.nkyE:self.nky_seq][::-1]
            E_ky = E_ky / self.deltaky
            E_ky = self.comm.allreduce(E_ky, op=MPI.SUM)

        elif not self.is_transposed:
            # Memory is shared along ky
            # In this case, self.dim_ky == 0 and self.dim_ky == 1
            # note that only the kx>=0 are in the spectral variables
            # to obtain the spectrum as a function of kx
            # we sum over all ky
            # the 2 is here because there are only the kx>=0

            raise NotImplementedError

        # E_kx = 2.*energy_fft.sum(self.dim_ky)/self.deltakx
        # E_kx[0] = E_kx[0]/2
        # E_kx = self.comm.allreduce(E_kx, op=MPI.SUM)
        # E_kx = E_kx[:self.nkxE]
        # # computation of E_ky
        # E_ky_tmp = energy_fft[:, 0].copy()
        # E_ky_tmp += 2*energy_fft[:, 1:].sum(1)
        # E_ky_tmp = np.ascontiguousarray(E_ky_tmp)
        # # print(self.rank, 'E_ky_tmp', E_ky_tmp, E_ky_tmp.shape)
        # E_ky_long = np.empty(self.nky_seq)
        # counts = self.comm.allgather(self.nky_loc)
        # self.comm.Allgatherv(sendbuf=[E_ky_tmp, MPI.DOUBLE],
        #                      recvbuf=[E_ky_long, (counts, None),
        #                               MPI.DOUBLE])
        # nkyE = self.nkyE
        # E_ky = E_ky_long[0:nkyE]
        # E_ky[1:nkyE] = E_ky[1:nkyE] + E_ky_long[self.nky_seq:nkyE:-1]
        # E_ky = E_ky/self.deltaky

        return E_kx, E_ky

    def compute_2dspectrum(self, E_fft):
        """Compute the 2D spectrum."""

        K = self.K

        nk0loc = self.shapeK_loc[0]
        nk1loc = self.shapeK_loc[1]

        # the copy is important: no *= !
        E_fft = 2 * E_fft

        if self.is_transposed:
            if self.rank == 0:
                E_fft[0, :] /= 2
            if (
                (self.is_sequential or self.rank == self.nb_proc - 1)
                and self.nx_seq % 2 == 0
                and self.shapeK_seq[0] == self.nkxE
            ):
                E_fft[-1, :] /= 2
        else:
            E_fft[:, 0] /= 2
            if self.nx_seq % 2 == 0:
                E_fft[:, -1] /= 2

        deltak = self.deltak

        khE = self.khE
        nkh = self.nkhE

        spectrum2D = np.zeros([nkh])
        for ik0 in range(nk0loc):
            for ik1 in range(nk1loc):
                E0D = E_fft[ik0, ik1]
                kappa0D = K[ik0, ik1]

                ikh = int(kappa0D / deltak)

                if ikh >= nkh - 1:
                    ikh = nkh - 1
                    spectrum2D[ikh] += E0D
                else:
                    coef_share = (kappa0D - khE[ikh]) / deltak
                    spectrum2D[ikh] += (1 - coef_share) * E0D
                    spectrum2D[ikh + 1] += coef_share * E0D

        if not self.is_sequential:
            spectrum2D = self.comm.allreduce(spectrum2D, op=mpi.MPI.SUM)
        return spectrum2D / deltak

    def compute_spectrum_kykx(self, energy_fft):
        """Compute a spectrum vs ky, kx. Return a dictionary."""
        if not self.is_transposed:
            # Memory is not shared
            # In this case, self.dim_ky == 0 and self.dim_kx == 1
            # note : only the kx >= 0 and ky>=0 are in the spectral variables
            E_kykxtmp = 2. * energy_fft / (self.deltakx * self.deltaky)
        else:
            # Memory is not shared
            # In this case, self.dim_ky == 1 and self.dim_kx == 0
            # note : only the kx >= 0 are in the spectral variables
            E_kykxtmp = 2. * np.transpose(energy_fft) / (
                self.deltakx * self.deltaky
            )
        # print(self.seq_indices_first_K)
        if self.is_sequential:
            #
            # computation of E_kykx
            E_kykxtmp[:, 0] = E_kykxtmp[:, 0] / 2

            if self.nx_seq % 2 == 0 and self.shapeK_seq[self.dim_kx] == self.nkxE:
                E_kykxtmp[:, -1] = E_kykxtmp[:, -1] / 2
            E_kykx = np.zeros([self.nkyE, self.nkxE])
            E_kykx[:self.nkyE, :self.nkxE] = E_kykxtmp[:self.nkyE, :self.nkxE]
            E_kykx[1:self.nkyE2, :] += E_kykxtmp[self.nkyE:self.nky_seq, :][::-1]
        elif self.is_transposed:
            # computation of E_kykx
            E_kykx_loc = E_kykxtmp

            if self.rank == 0:
                E_kykx_loc[:, 0] = E_kykx_loc[:, 0] / 2

            if (
                self.rank == self.nb_proc - 1
                and self.nx_seq % 2 == 0
                and self.shapeK_seq[0] == self.nkxE
            ):
                E_kykx_loc[:, -1] = E_kykx_loc[:, -1] / 2

            E_kykx = np.zeros([self.nkyE, self.nkxE])
            nkx_start = self.seq_indices_first_K[0]
            E_kykx[:, nkx_start:self.nkx_loc + nkx_start] = E_kykx_loc[
                :self.nkyE, :self.nkx_loc
            ]

            E_kykx[
                1:self.nkyE2, nkx_start:self.nkx_loc + nkx_start
            ] += E_kykx_loc[
                self.nkyE:self.nky_seq, :self.nkx_loc
            ][
                ::-1
            ]
            E_kykx = self.comm.allreduce(E_kykx, op=MPI.SUM)

        elif not self.is_transposed:

            raise NotImplementedError

        return E_kykx

    def projection_perp(self, fx_fft, fy_fft):
        """Project (inplace) a vector perpendicular to the wavevector.

        The resulting vector is divergence-free.

        """
        KX = self.KX
        KY = self.KY
        a = fx_fft - self.KX_over_K2 * (KX * fx_fft + KY * fy_fft)
        b = fy_fft - self.KY_over_K2 * (KX * fx_fft + KY * fy_fft)
        fx_fft[:] = a
        fy_fft[:] = b
        return a, b

    def rotfft_from_vecfft(self, vecx_fft, vecy_fft):
        """Return the rotational of a vector in spectral space."""
        return rotfft_from_vecfft(vecx_fft, vecy_fft, self.KX, self.KY)

    def divfft_from_vecfft(self, vecx_fft, vecy_fft):
        """Return the divergence of a vector in spectral space."""
        return divfft_from_vecfft(vecx_fft, vecy_fft, self.KX, self.KY)

    def vecfft_from_rotfft(self, rot_fft):
        """Return the velocity in spectral space computed from the
        rotational."""
        return vecfft_from_rotfft(rot_fft, self.KX_over_K2, self.KY_over_K2)

    def vecfft_from_divfft(self, div_fft):
        """Return the velocity in spectral space computed from the
        divergence."""
        return vecfft_from_divfft(div_fft, self.KX_over_K2, self.KY_over_K2)

    def gradfft_from_fft(self, f_fft):
        """Return the gradient of f_fft in spectral space."""
        return gradfft_from_fft(f_fft, self.KX, self.KY)

    def dealiasing_variable(self, f_fft):
        """Dealiasing a variable."""
        if self._has_to_dealiase:
            dealiasing_variable(
                f_fft, self.where_dealiased, self.nK0_loc, self.nK1_loc
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


# if __name__ == '__main__':
#     self = OperatorsPseudoSpectral2D(5, 3, 2*pi, 1*pi)

#     a = np.random.random(self.opfft.get_local_size_X()).reshape(
#         self.opfft.get_shapeX_loc())
#     afft = self.fft(a)
#     a = self.ifft(afft)
#     afft = self.fft(a)

#     print('energy spatial C:', self.compute_energy_from_X(a))
#     print('energy fft      :', self.compute_energy_from_K(afft))

#     print('energy spatial P:', (a**2).mean()/2)

#     energy_fft = 0.5 * abs(afft)**2

#     E_kx, E_ky = self.compute_1dspectra(energy_fft)

#     print('energy E_kx     ;', E_kx.sum()*self.deltakx)
#     print('energy E_ky     :', E_ky.sum()*self.deltaky)

#     E_kh = self.compute_2dspectrum(energy_fft)

#     print('energy E_kh     :', E_kh.sum()*self.deltak)
