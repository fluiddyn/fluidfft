
from __future__ import print_function

from past.builtins import basestring

from math import pi

import numpy as np

from fluiddyn.util import mpi
from fluiddyn.util.easypyfft import FFTW3DReal2Complex

from fluidfft import create_fft_object
from fluidfft.fft2d.operators import _make_str_length

from .util_pythran import (
    project_perpk3d)

from .dream_pythran import _vgradv_from_v2

if mpi.nb_proc > 1:
    MPI = mpi.MPI


class OperatorsPseudoSpectral3D(object):

    def __init__(self, nx, ny, nz, lx, ly, lz, fft='fft3d.with_fftw2d',
                 coef_dealiasing=1.):
        self.nx = self.nx_seq = nx
        self.ny = self.ny_seq = ny
        self.nz = self.nz_seq = nz

        if isinstance(fft, basestring):
            if fft.lower() == 'fftwpy':
                op_fft = FFTW3DReal2Complex(nx, ny, nz)
            elif any([fft.startswith(s) for s in ['fluidfft.', 'fft3d.']]):
                op_fft = create_fft_object(fft, nz, ny, nx)
            else:
                raise ValueError
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

        self.ifft3d = op_fft.ifft
        self.fft3d = op_fft.fft
        self.sum_wavenumbers = op_fft.sum_wavenumbers
        self.compute_energy_from_X = op_fft.compute_energy_from_X
        self.compute_energy_from_K = op_fft.compute_energy_from_K

        self.shapeK_loc = op_fft.get_shapeK_loc()
        self.nk0, self.nk1, self.nk2 = self.shapeK_loc

        order = op_fft.get_dimX_K()
        if order == (0, 1, 2):
            self.deltaks = deltakz, deltaky, deltakx
        elif order == (1, 0, 2):
            self.deltaks = deltaky, deltakz, deltakx
        elif order == (2, 1, 0):
            self.deltaks = deltakx, deltaky, deltakz
        else:
            print('order =', order)
            raise NotImplementedError

        k0_adim_loc, k1_adim_loc, k2_adim_loc = op_fft.get_k_adim_loc()

        self.k0 = self.deltaks[0] * k0_adim_loc
        self.k1 = self.deltaks[1] * k1_adim_loc
        self.k2 = self.deltaks[2] * k2_adim_loc

        # oh that's strange!
        K1, K0, K2 = np.meshgrid(self.k1, self.k0, self.k2, copy=False)

        self.Kz = K0
        self.Ky = K1
        self.Kx = K2

        self.K2 = K0**2 + K1**2 + K2**2
        self.K8 = self.K2**4

        self.seq_index_firstK0, self.seq_index_firstK1 = \
            op_fft.get_seq_indices_first_K()

        self.K_square_nozero = self.K2.copy()

        if self.seq_index_firstK0 == 0 and self.seq_index_firstK1 == 0:
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

    def produce_str_describing_grid(self):
        return '{}x{}x{}'.format(self.nx_seq, self.ny_seq, self.nz_seq)

    def produce_str_describing_oper(self):
        """Produce a string describing the operator."""
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

    def constant_arrayX(self, value=None, SHAPE='LOC'):
        """Return a constant array in real space."""
        if SHAPE == 'LOC':
            shapeX = self.shapeX_loc
        elif SHAPE == 'SEQ':
            shapeX = self.shapeX_seq
        else:
            raise ValueError('SHAPE should be "LOC" of "SEQ"')
        if value is None:
            field = np.empty(shapeX)
        elif value == 0:
            field = np.zeros(shapeX)
        else:
            field = value*np.ones(shapeX)
        return field

    def project_perpk3d(self, vx_fft, vy_fft, vz_fft):
        project_perpk3d(vx_fft, vy_fft, vz_fft, self.Kx, self.Ky, self.Kz,
                        self.K_square_nozero)

    def vgradv_from_v(self, vx, vy, vz,
                      vx_fft=None, vy_fft=None, vz_fft=None):

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

        if vx_fft is None:
            # function(float64[][][]) -> complex128[][][]
            fft3d = self.fft3d
            vx_fft = fft3d(vx)
            vy_fft = fft3d(vy)
            vz_fft = fft3d(vz)

        return _vgradv_from_v2(vx, vy, vz, vx_fft, vy_fft, vz_fft,
                               self.Kx, self.Ky, self.Kz, self.ifft3d)
