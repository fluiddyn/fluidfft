"""Operators 3d (:mod:`fluidfft.fft3d.operators`)
=================================================

.. autoclass:: OperatorsPseudoSpectral3D
   :members:
   :undoc-members:

"""

from math import pi

import numpy as np

from transonic import boost, Array, Type
from fluiddyn.util import mpi

from fluidfft import create_fft_object, import_fft_class
from fluidfft.base import OperatorsBase
from fluidfft.fft2d.operators import _make_str_length

Ac = "complex128[:,:,:]"
Af = "float64[:,:,:]"
A = Array[Type(np.float64, np.complex128), "3d"]


@boost
def vector_product(ax: Af, ay: Af, az: Af, bx: Af, by: Af, bz: Af):
    """Compute the vector product.

    Warning: the arrays bx, by, bz are overwritten.

    """
    n0, n1, n2 = ax.shape

    for i0 in range(n0):
        for i1 in range(n1):
            for i2 in range(n2):
                elem_ax = ax[i0, i1, i2]
                elem_ay = ay[i0, i1, i2]
                elem_az = az[i0, i1, i2]
                elem_bx = bx[i0, i1, i2]
                elem_by = by[i0, i1, i2]
                elem_bz = bz[i0, i1, i2]

                bx[i0, i1, i2] = elem_ay * elem_bz - elem_az * elem_by
                by[i0, i1, i2] = elem_az * elem_bx - elem_ax * elem_bz
                bz[i0, i1, i2] = elem_ax * elem_by - elem_ay * elem_bx

    return bx, by, bz


@boost
def loop_spectra3d(spectrum_k0k1k2: Af, ks: "float[]", K2: Af):
    """Compute the 3d spectrum."""
    deltak = ks[1]
    nk = len(ks)
    spectrum3d = np.zeros(nk)
    nk0, nk1, nk2 = spectrum_k0k1k2.shape
    for ik0 in range(nk0):
        for ik1 in range(nk1):
            for ik2 in range(nk2):
                value = spectrum_k0k1k2[ik0, ik1, ik2]
                kappa = np.sqrt(K2[ik0, ik1, ik2])
                ik = int(kappa / deltak)
                if ik >= nk - 1:
                    ik = nk - 1
                    spectrum3d[ik] += value
                else:
                    coef_share = (kappa - ks[ik]) / deltak
                    spectrum3d[ik] += (1 - coef_share) * value
                    spectrum3d[ik + 1] += coef_share * value

    return spectrum3d


@boost
def loop_spectra_kzkh(
    spectrum_k0k1k2: Af, khs: "float[]", KH: Af, kzs: "float[]", KZ: Af
):
    """Compute the kz-kh spectrum."""
    deltakh = khs[1]
    deltakz = kzs[1]
    nkh = len(khs)
    nkz = len(kzs)
    spectrum_kzkh = np.zeros((nkz, nkh))
    nk0, nk1, nk2 = spectrum_k0k1k2.shape
    for ik0 in range(nk0):
        for ik1 in range(nk1):
            for ik2 in range(nk2):
                value = spectrum_k0k1k2[ik0, ik1, ik2]
                kappa = KH[ik0, ik1, ik2]
                ikh = int(kappa / deltakh)
                kz = abs(KZ[ik0, ik1, ik2])
                ikz = int(round(kz / deltakz))
                if ikz >= nkz - 1:
                    ikz = nkz - 1
                if ikh >= nkh - 1:
                    ikh = nkh - 1
                    spectrum_kzkh[ikz, ikh] += value
                else:
                    coef_share = (kappa - khs[ikh]) / deltakh
                    spectrum_kzkh[ikz, ikh] += (1 - coef_share) * value
                    spectrum_kzkh[ikz, ikh + 1] += coef_share * value
    return spectrum_kzkh


def get_simple_3d_seq_method():
    """Get a simple 3d sequential method"""
    try:
        import pyfftw

    except ImportError:
        return "fft3d.with_fftw3d"
    else:
        del pyfftw
        return "fft3d.with_pyfftw"


def get_simple_3d_mpi_method():
    """Get a simple 3d parallel method"""
    method = "fft3d.mpi_with_fftwmpi3d"
    try:
        import_fft_class(method)
    except (ImportError, ValueError):
        method = "fft3d.mpi_with_fftw1d"
    return method


@boost
class OperatorsPseudoSpectral3D(OperatorsBase):
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

    Kx: Af
    Ky: Af
    Kz: Af
    inv_K_square_nozero: Af

    def __init__(self, nx, ny, nz, lx, ly, lz, fft=None, coef_dealiasing=1.0):
        self.nx = self.nx_seq = nx
        self.ny = self.ny_seq = ny
        self.nz = self.nz_seq = nz

        if fft is None or fft == "default":
            if mpi.nb_proc == 1:
                fft = get_simple_3d_seq_method()
            else:
                fft = get_simple_3d_mpi_method()

        if isinstance(fft, str):
            if fft.lower() == "sequential":
                fft = get_simple_3d_seq_method()
            elif fft.lower() == "fftwpy":
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

        self.oper_fft = self._op_fft = op_fft
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

        self.K2 = K0**2 + K1**2 + K2**2
        self.K8 = self.K2**4

        self.is_sequential = op_fft.get_shapeK_loc() == op_fft.get_shapeK_seq()

        self.seq_indices_first_K = op_fft.get_seq_indices_first_K()
        self.seq_indices_first_X = op_fft.get_seq_indices_first_X()

        K_square_nozero = self.K2.copy()

        if all(index == 0 for index in self.seq_indices_first_K):
            K_square_nozero[0, 0, 0] = 1e-14

        self.inv_K_square_nozero = 1.0 / K_square_nozero

        self.coef_dealiasing = coef_dealiasing

        CONDKX = abs(self.Kx) >= self.coef_dealiasing * deltakx * nx / 2
        CONDKY = abs(self.Ky) >= self.coef_dealiasing * deltaky * ny / 2
        CONDKZ = abs(self.Kz) >= self.coef_dealiasing * deltakz * nz / 2
        where_dealiased = CONDKX | CONDKY | CONDKZ
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

        self.deltakh = max(self.deltakx, self.deltaky)
        self.khmax_spectra = min(self.kxmax_spectra, self.kymax_spectra)
        self.nkh_spectra = max(2, int(self.khmax_spectra / self.deltakh))
        self.kh_spectra = self.deltakh * np.arange(self.nkh_spectra)

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
        return self._op_fft.create_arrayX(value, shapeX)

    def create_arrayK(self, value=None, shape="loc"):
        """Return a constant array in spectral space."""
        shapeK = self._get_shapeK(shape)
        return self._op_fft.create_arrayK(value, shapeK)

    def create_arrayX_random(self, shape="loc", min_val=None, max_val=None):
        """Return a random array in real space."""
        shape = self._get_shapeX(shape)
        values = np.random.random(shape)
        return self._rescale_random(values, min_val, max_val)

    def create_arrayK_random(self, shape="loc", min_val=None, max_val=None):
        """Return a random array in real space."""
        shape = self._get_shapeK(shape)
        values = np.random.random(shape) + 1j * np.random.random(shape)
        return self._rescale_random(values, min_val, max_val)

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

    @boost
    def project_perpk3d_noloop(self, vx_fft: A, vy_fft: A, vz_fft: A):
        """Project (inplace) a vector perpendicular to the wavevector.

        The resulting vector is divergence-free.

        """
        # function important for the performance of 3d fluidsim solvers
        tmp = (
            self.Kx * vx_fft + self.Ky * vy_fft + self.Kz * vz_fft
        ) * self.inv_K_square_nozero

        vx_fft -= self.Kx * tmp
        vy_fft -= self.Ky * tmp
        vz_fft -= self.Kz * tmp

    @boost
    def project_perpk3d(self, vx_fft: A, vy_fft: A, vz_fft: A):
        """Project (inplace) a vector perpendicular to the wavevector.

        The resulting vector is divergence-free.

        """
        # function important for the performance of 3d fluidsim solvers
        # this version with loop is really faster than `project_perpk3d_noloop`
        n0, n1, n2 = vx_fft.shape
        for i0 in range(n0):
            for i1 in range(n1):
                for i2 in range(n2):
                    tmp = (
                        self.Kx[i0, i1, i2] * vx_fft[i0, i1, i2]
                        + self.Ky[i0, i1, i2] * vy_fft[i0, i1, i2]
                        + self.Kz[i0, i1, i2] * vz_fft[i0, i1, i2]
                    ) * self.inv_K_square_nozero[i0, i1, i2]

                    vx_fft[i0, i1, i2] -= self.Kx[i0, i1, i2] * tmp
                    vy_fft[i0, i1, i2] -= self.Ky[i0, i1, i2] * tmp
                    vz_fft[i0, i1, i2] -= self.Kz[i0, i1, i2] * tmp

    @boost
    def divfft_from_vecfft(self, vx_fft: Ac, vy_fft: Ac, vz_fft: Ac):
        """Return the divergence of a vector in spectral space."""
        return 1j * (self.Kx * vx_fft + self.Ky * vy_fft + self.Kz * vz_fft)

    @boost
    def rotfft_from_vecfft(self, vx_fft: Ac, vy_fft: Ac, vz_fft: Ac):
        """Return the curl of a vector in spectral space."""

        return (
            1j * (self.Ky * vz_fft - self.Kz * vy_fft),
            1j * (self.Kz * vx_fft - self.Kx * vz_fft),
            1j * (self.Kx * vy_fft - self.Ky * vx_fft),
        )

    @boost
    def rotfft_from_vecfft_outin(
        self,
        vx_fft: Ac,
        vy_fft: Ac,
        vz_fft: Ac,
        rotxfft: Ac,
        rotyfft: Ac,
        rotzfft: Ac,
    ):
        """Return the curl of a vector in spectral space."""
        # function important for the performance of 3d fluidsim solvers

        # cleaner but slightly slower
        # rotxfft[:] = 1j * (self.Ky * vz_fft - self.Kz * vy_fft)
        # rotyfft[:] = 1j * (self.Kz * vx_fft - self.Kx * vz_fft)
        # rotzfft[:] = 1j * (self.Kx * vy_fft - self.Ky * vx_fft)

        # seems faster (at least for small cases)
        n0, n1, n2 = vx_fft.shape
        for i0 in range(n0):
            for i1 in range(n1):
                for i2 in range(n2):
                    rotxfft[i0, i1, i2] = 1j * (
                        self.Ky[i0, i1, i2] * vz_fft[i0, i1, i2]
                        - self.Kz[i0, i1, i2] * vy_fft[i0, i1, i2]
                    )
                    rotyfft[i0, i1, i2] = 1j * (
                        self.Kz[i0, i1, i2] * vx_fft[i0, i1, i2]
                        - self.Kx[i0, i1, i2] * vz_fft[i0, i1, i2]
                    )
                    rotzfft[i0, i1, i2] = 1j * (
                        self.Kx[i0, i1, i2] * vy_fft[i0, i1, i2]
                        - self.Ky[i0, i1, i2] * vx_fft[i0, i1, i2]
                    )

    def div_vb_fft_from_vb(self, vx, vy, vz, b):
        r"""Compute :math:`\nabla \cdot (\boldsymbol{v} b)` in spectral space."""
        fft3d = self.fft3d

        vxbfft = fft3d(vx * b)
        vybfft = fft3d(vy * b)
        vzbfft = fft3d(vz * b)

        return self.divfft_from_vecfft(vxbfft, vybfft, vzbfft)

    @boost
    def rotzfft_from_vxvyfft(self, vx_fft: Ac, vy_fft: Ac):
        """Compute the z component of the curl in spectral space."""
        return 1j * (self.Kx * vy_fft - self.Ky * vx_fft)

    def get_XYZ_loc(self):
        """Compute the local 3d arrays with the x, y, and y values."""

        if self.shapeX_seq != self.shapeX_loc:
            i0_seq_start, i1_seq_start, i2_seq_start = self.seq_indices_first_X
            if self.shapeX_seq[1:] != self.shapeX_loc[1:]:
                # general solution
                # mpi.print_sorted(
                #     'in get_XYZ_loc:',
                #     '(i0_seq_start, i1_seq_start, i2_seq_start):',
                #     (i0_seq_start, i1_seq_start, i2_seq_start))

                z_loc = self.z_seq[
                    i0_seq_start : i0_seq_start + self.shapeX_loc[0]
                ]
                y_loc = self.y_seq[
                    i1_seq_start : i1_seq_start + self.shapeX_loc[1]
                ]
                x_loc = self.x_seq[
                    i2_seq_start : i2_seq_start + self.shapeX_loc[2]
                ]

            # mpi.print_sorted('z_loc', z_loc)
            # mpi.print_sorted('y_loc', y_loc)
            # mpi.print_sorted('x_loc', x_loc)

            else:
                # 1d decomposition
                x_loc = self.x_seq
                y_loc = self.y_seq
                z_loc = self.z_seq[
                    i0_seq_start : i0_seq_start + self.shapeX_loc[0]
                ]
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
                    spectrum_tmp_seq[istart : istart + nk_loc] = spectrum_tmp_loc
                    spectrum_ki = spectrum_tmp_seq[:nk_spectra]
                    nk1 = (ni + 1) // 2
                    spectrum_ki[1:nk1] += spectrum_tmp_seq[nk_spectra:][::-1]
                else:
                    spectrum_tmp_seq = np.zeros(nk_spectra)
                    spectrum_tmp_seq[istart : istart + nk_loc] = spectrum_tmp_loc
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

    def compute_spectrum_kzkh(self, energy_fft):
        """Compute the kz-kh spectrum."""
        khs = self.kh_spectra
        KH = np.sqrt(self.Kx**2 + self.Ky**2)
        kzs = self.deltakz * np.arange(self.nkz_spectra)
        spectrum_k0k1k2 = self._compute_spectrum3d_loc(energy_fft)
        spectrum = loop_spectra_kzkh(spectrum_k0k1k2, khs, KH, kzs, self.Kz)
        if self._is_mpi_lib:
            spectrum = mpi.comm.allreduce(spectrum, op=mpi.MPI.SUM)
        return spectrum / (self.deltakz * self.deltakh)

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
