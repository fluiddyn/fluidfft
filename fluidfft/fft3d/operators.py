"""Operators 3d (:mod:`fluidfft.fft3d.operators`)
=================================================

.. autoclass:: OperatorsPseudoSpectral3D
   :members:
   :undoc-members:

"""
from __future__ import print_function

from past.builtins import basestring

from math import pi

import numpy as np

from fluiddyn.util import mpi
from fluiddyn.calcul.easypyfft import FFTW3DReal2Complex

from fluidfft import create_fft_object
from fluidfft.fft2d.operators import _make_str_length

from .util_pythran import (
    project_perpk3d, divfft_from_vecfft, rotfft_from_vecfft, vector_product)

from .dream_pythran import _vgradv_from_v2

if mpi.nb_proc > 1:
    MPI = mpi.MPI


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
    def __init__(self, nx, ny, nz, lx, ly, lz, fft=None,
                 coef_dealiasing=1.):
        self.nx = self.nx_seq = nx
        self.ny = self.ny_seq = ny
        self.nz = self.nz_seq = nz

        if fft is None:
            if mpi.nb_proc == 1:
                fft = 'fft3d.with_fftw3d'
            else:
                fft = 'fft3d.mpi_with_fftw1d'

        if isinstance(fft, basestring):
            if fft.lower() == 'fftwpy':
                op_fft = FFTW3DReal2Complex(nx, ny, nz)
            elif any([fft.startswith(s) for s in ['fluidfft.', 'fft3d.']]):
                op_fft = create_fft_object(fft, nz, ny, nx)
            else:
                raise ValueError(
                    "Cannot instantiate {}.".format(fft) +
                    " Expected something like 'fftwpy'"
                    " or 'fluidfft.fft3d.<method>' or 'fft3d.<method>'")
        elif isinstance(fft, type):
            op_fft = fft(nz, ny, nx)
        else:
            op_fft = fft

        self._op_fft = op_fft
        self.type_fft = op_fft.__class__.__module__

        self.shapeX_seq = op_fft.get_shapeX_seq()
        self.shapeX_loc = op_fft.get_shapeX_loc()

        Lx = self.Lx = float(lx)
        Ly = self.Ly = float(ly)
        Lz = self.Lz = float(lz)

        self.deltax = Lx / nx
        self.deltay = Ly / ny
        self.deltaz = Lz / nz

        self.x_seq = self.deltax*np.arange(nx)
        self.y_seq = self.deltay*np.arange(ny)
        self.z_seq = self.deltaz*np.arange(nz)

        deltakx = 2*pi/Lx
        deltaky = 2*pi/Ly
        deltakz = 2*pi/Lz

        self.ifft = self.ifft3d = op_fft.ifft
        self.fft = self.fft3d = op_fft.fft

        self.ifft_as_arg = op_fft.ifft_as_arg
        self.fft_as_arg = op_fft.fft_as_arg

        self.sum_wavenumbers = op_fft.sum_wavenumbers
        self.compute_energy_from_X = op_fft.compute_energy_from_X
        self.compute_energy_from_K = op_fft.compute_energy_from_K

        self.shapeK = self.shapeK_loc = op_fft.get_shapeK_loc()
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
            print('order =', order)
            raise NotImplementedError

        k0_adim_loc, k1_adim_loc, k2_adim_loc = op_fft.get_k_adim_loc()

        self.k0 = self.deltaks[0] * k0_adim_loc
        self.k1 = self.deltaks[1] * k1_adim_loc
        self.k2 = self.deltaks[2] * k2_adim_loc

        # oh that's strange!
        K1, K0, K2 = np.meshgrid(self.k1, self.k0, self.k2, copy=False)

        K0 = np.ascontiguousarray(K0)
        K1 = np.ascontiguousarray(K1)
        K2 = np.ascontiguousarray(K2)

        assert K0.shape == self.shapeK_loc

        self.Kz = K0
        self.Ky = K1
        self.Kx = K2

        self.K2 = K0**2 + K1**2 + K2**2
        self.K8 = self.K2**4

        self.seq_indices_first_K = op_fft.get_seq_indices_first_K()
        # self.seq_indices_first_X = op_fft.get_seq_indices_first_X()

        self.K_square_nozero = self.K2.copy()

        if all(index == 0 for index in self.seq_indices_first_K):
            self.K_square_nozero[0, 0, 0] = 1e-14

        self.coef_dealiasing = coef_dealiasing

        CONDKX = abs(self.Kx) > self.coef_dealiasing*self.k2.max()
        CONDKY = abs(self.Ky) > self.coef_dealiasing*self.k1.max()
        CONDKZ = abs(self.Kz) > self.coef_dealiasing*self.k0.max()
        where_dealiased = np.logical_or(CONDKX, CONDKY, CONDKZ)
        self.where_dealiased = np.array(where_dealiased, dtype=np.int8)
        if mpi.nb_proc > 1:
            self.gather_Xspace = op_fft.gather_Xspace
            self.scatter_Xspace = op_fft.scatter_Xspace

        self.rank = mpi.rank

        self.tmp_fields_fft = tuple(self.constant_arrayK() for n in range(6))

    def produce_str_describing_grid(self):
        """Produce a short string describing the grid."""
        return '{}x{}x{}'.format(self.nx_seq, self.ny_seq, self.nz_seq)

    def produce_str_describing_oper(self):
        """Produce a short string describing the operator."""
        str_Lx = _make_str_length(self.Lx)
        str_Ly = _make_str_length(self.Ly)
        str_Lz = _make_str_length(self.Lz)

        return ('{}x{}x{}_V' + str_Lx + 'x' + str_Ly +
                'x' + str_Lz).format(self.nx_seq, self.ny_seq, self.nz_seq)

    def produce_long_str_describing_oper(self):
        """Produce a string describing the operator."""

        str_Lx = _make_str_length(self.Lx)
        str_Ly = _make_str_length(self.Ly)
        str_Lz = _make_str_length(self.Lz)

        return (
            'type fft: ' + self.type_fft + '\n' +
            'nx = {0:6d} ; ny = {1:6d}\n'.format(self.nx_seq, self.ny_seq) +
            'Lx = ' + str_Lx + ' ; Ly = ' + str_Ly +
            ' ; Lz = ' + str_Lz + '\n')

    def _get_shapeX(self, shape='loc'):
        if shape.lower() == 'loc':
            return self.shapeX_loc
        elif shape.lower() == 'seq':
            return self.shapeX_seq
        else:
            raise ValueError('shape should be "loc" or "seq"')

    def _get_shapeK(self, shape='loc'):
        if shape.lower() == 'loc':
            return self.shapeK_loc
        elif shape.lower() == 'seq':
            return self.shapeK_seq
        else:
            raise ValueError('shape should be "loc" or "seq"')

    def constant_arrayX(self, value=None, shape='loc'):
        """Return a constant array in real space."""
        shapeX = self._get_shapeX(shape)
        if value is None:
            field = np.empty(shapeX)
        elif value == 0:
            field = np.zeros(shapeX)
        else:
            field = value*np.ones(shapeX)
        return field

    def constant_arrayK(self, value=None, shape='loc'):
        """Return a constant array in real space."""
        shapeK = self._get_shapeK(shape)
        if value is None:
            field = np.empty(shapeK, dtype=np.complex128)
        elif value == 0:
            field = np.zeros(shapeK, dtype=np.complex128)
        else:
            field = value*np.ones(shapeK, dtype=np.complex128)
        return field

    def random_arrayX(self, shape='loc'):
        """Return a random array in real space."""
        shapeX = self._get_shapeX(shape)
        return np.random.random(shapeX)

    def sum_wavenumbers_versatile(self, field_fft):
        """Compute the sum over all wavenumbers (versatile version).

        This function should return the same result than
        :func:`sum_wavenumbers`.

        It is here mainly to check that the classes are well implemented.
        """
        raise NotImplementedError

    def project_perpk3d(self, vx_fft, vy_fft, vz_fft):
        """Project (inplace) a vector perpendicular to the wavevector.

        The resulting vector is divergence-free.

        """
        project_perpk3d(vx_fft, vy_fft, vz_fft, self.Kx, self.Ky, self.Kz,
                        self.K_square_nozero)

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
    
    def div_vv_fft_from_v(self, vx, vy, vz):
        r"""Compute :math:`\nabla \cdot (\boldsymbol{v} \boldsymbol{v})` in
        spectral space.

        """
        # function(float64[][][]) -> complex128[][][]
        # fft3d = self.fft3d

        # vxvxfft = fft3d(vx*vx)
        # vyvyfft = fft3d(vy*vy)
        # vzvzfft = fft3d(vz*vz)

        # vxvyfft = vyvxfft = fft3d(vx*vy)
        # vxvzfft = vzvxfft = fft3d(vx*vz)
        # vyvzfft = vzvyfft = fft3d(vy*vz)

        vxvxfft = self.tmp_fields_fft[0]
        vyvyfft = self.tmp_fields_fft[1]
        vzvzfft = self.tmp_fields_fft[2]

        vxvyfft = vyvxfft = self.tmp_fields_fft[3]
        vxvzfft = vzvxfft = self.tmp_fields_fft[4]
        vyvzfft = vzvyfft = self.tmp_fields_fft[5]

        fft_as_arg = self.fft_as_arg

        fft_as_arg(vx*vx, vxvxfft)
        fft_as_arg(vy*vy, vyvyfft)
        fft_as_arg(vz*vz, vzvzfft)

        fft_as_arg(vx*vy, vxvyfft)
        fft_as_arg(vx*vz, vxvzfft)
        fft_as_arg(vy*vz, vyvzfft)

        # float64[][][]
        Kx = self.Kx
        Ky = self.Ky
        Kz = self.Kz

        return (divfft_from_vecfft(vxvxfft, vyvxfft, vzvxfft, Kx, Ky, Kz),
                divfft_from_vecfft(vxvyfft, vyvyfft, vzvyfft, Kx, Ky, Kz),
                divfft_from_vecfft(vxvzfft, vyvzfft, vzvzfft, Kx, Ky, Kz))

    def div_vb_fft_from_vb(self, vx, vy, vz, b):
        r"""Compute :math:`\nabla \cdot (\boldsymbol{v} b)` in spectral space.

        """
        fft3d = self.fft3d

        vxbfft = fft3d(vx*b)
        vybfft = fft3d(vy*b)
        vzbfft = fft3d(vz*b)

        return divfft_from_vecfft(vxbfft, vybfft, vzbfft,
                                  self.Kx, self.Ky, self.Kz)

    def vgradv_from_v(self, vx, vy, vz,
                      vx_fft=None, vy_fft=None, vz_fft=None):
        r"""Compute :math:`\boldsymbol{v} \cdot \nabla \boldsymbol{v}` in
        real space.

        """
        if vx_fft is None:
            # function(float64[][][]) -> complex128[][][]
            fft3d = self.fft3d
            vx_fft = fft3d(vx)
            vy_fft = fft3d(vy)
            vz_fft = fft3d(vz)

        # function(complex128[][][]) -> float64[][][]
        ifft3d = self.ifft3d

        # float64[][][]
        Kx = self.Kx
        Ky = self.Ky
        Kz = self.Kz

        px_vx_fft = 1j * Kx * vx_fft
        py_vx_fft = 1j * Ky * vx_fft
        pz_vx_fft = 1j * Kz * vx_fft

        px_vy_fft = 1j * Kx * vy_fft
        py_vy_fft = 1j * Ky * vy_fft
        pz_vy_fft = 1j * Kz * vy_fft

        px_vz_fft = 1j * Kx * vz_fft
        py_vz_fft = 1j * Ky * vz_fft
        pz_vz_fft = 1j * Kz * vz_fft

        vgradvx = (vx * ifft3d(px_vx_fft) +
                   vy * ifft3d(py_vx_fft) +
                   vz * ifft3d(pz_vx_fft))

        vgradvy = (vx * ifft3d(px_vy_fft) +
                   vy * ifft3d(py_vy_fft) +
                   vz * ifft3d(pz_vy_fft))

        vgradvz = (vx * ifft3d(px_vz_fft) +
                   vy * ifft3d(py_vz_fft) +
                   vz * ifft3d(pz_vz_fft))

        return vgradvx, vgradvy, vgradvz

    def vgradv_from_v2(self, vx, vy, vz,
                       vx_fft=None, vy_fft=None, vz_fft=None):
        r"""Compute :math:`\boldsymbol{v} \cdot \nabla \boldsymbol{v}` in
        real space.

        """
        if vx_fft is None:
            # function(float64[][][]) -> complex128[][][]
            fft3d = self.fft3d
            vx_fft = fft3d(vx)
            vy_fft = fft3d(vy)
            vz_fft = fft3d(vz)

        return _vgradv_from_v2(vx, vy, vz, vx_fft, vy_fft, vz_fft,
                               self.Kx, self.Ky, self.Kz, self.ifft3d)

    def rotzfft_from_vxvyfft(self, vx_fft, vy_fft):
        """Compute the z component of the curl in spectral space."""
        return 1j * (self.Kx * vy_fft - self.Ky * vx_fft)

    def get_XYZ_loc(self):
        """Compute the local 3d arrays with the x, y, and y values.

        The implementation of this function is not easy for some classes...  We
        need a not-implemented function :func:`get_seq_indices_first_X`...

        """

        if self.shapeX_seq != self.shapeX_loc:

            all_shape_loc = np.empty((mpi.nb_proc, 3), dtype=int)
            mpi.comm.Allgather(np.array(self.shapeX_loc), all_shape_loc)

            if self.shapeX_seq[1:] != self.shapeX_loc[1:]:
                # in this case, it it complicated...
                raise NotImplementedError

            else:
                # 1d decomposition
                x_loc = self.x_seq
                y_loc = self.y_seq

                # actually with 1d decomposition we need only this
                all_shape0_loc = all_shape_loc[:, 0]
                i0_seq_start = sum(all_shape0_loc[:mpi.rank])

                z_loc = self.z_seq[
                    i0_seq_start:i0_seq_start+self.shapeX_loc[0]]
        else:
            x_loc = self.x_seq
            y_loc = self.y_seq
            z_loc = self.z_seq

        Y, Z, X = np.meshgrid(y_loc, z_loc, x_loc, copy=False)

        assert X.shape == Y.shape == Z.shape == self.shapeX_loc

        return X, Y, Z

    def compute_1dspectra(self, energy_fft):
        """Compute the 1D spectra.

        NotImplemented!

        Returns
        -------

        E_kx

        E_ky

        E_kz

        """
        raise NotImplementedError

    def compute_3dspectrum(self, energy_fft):
        """Compute the 3D spectrum.

        NotImplemented!

        """
        raise NotImplementedError

    def compute_spectra_2vars(self, energy_fft):
        """Compute spectra vs 2 variables.

        NotImplemented!

        Returns
        -------

        E_kx_kyz

        E_ky_kzx

        E_kz_kxy

        """
        raise NotImplementedError
