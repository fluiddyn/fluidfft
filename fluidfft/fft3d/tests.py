
from __future__ import print_function, division

import unittest
from math import pi

import numpy as np

from fluiddyn.util import mpi

from fluidfft.fft3d import get_classes_seq, get_classes_mpi
from fluidfft.fft3d.operators import OperatorsPseudoSpectral3D

n = 8

rank = mpi.rank
nb_proc = mpi.nb_proc

classes_seq = get_classes_seq()
classes_seq = {name: cls for name, cls in classes_seq.items()
               if cls is not None}

if nb_proc > 1:
    classes_mpi = get_classes_mpi()
    classes_mpi = {name: cls for name, cls in classes_mpi.items()
                   if cls is not None}


def make_testop_functions(name, cls):

    tests = {}
    shapes = {'even': (4, 8, 12)}
    if nb_proc == 1:
        shapes['odd'] = (5, 3, 7)

    for key, (n0, n1, n2) in shapes.items():

        def test(self, n0=n0, n1=n1, n2=n2):
            op = OperatorsPseudoSpectral3D(n2, n1, n0,
                                           12, 8, 4, fft=cls)
            op_fft = op._op_fft

            op_fft.run_tests()

            a = np.random.random(
                op_fft.get_local_size_X()).reshape(
                    op_fft.get_shapeX_loc())
            a0 = a.copy()
            afft = op.fft3d(a)
            self.assertTrue(np.allclose(a, a0))
            afft0 = afft.copy()
            a = op.ifft3d(afft)
            self.assertTrue(np.allclose(afft, afft0))
            afft = op.fft3d(a)

            nrja = op.compute_energy_from_X(a)
            nrjafft = op.compute_energy_from_K(afft)
            self.assertAlmostEqual(nrja, nrjafft)

            energy_fft = 0.5 * abs(afft)**2
            nrj = op.sum_wavenumbers(energy_fft)
            self.assertAlmostEqual(nrjafft, nrj)

            try:
                nrj_versatile = op.sum_wavenumbers_versatile(energy_fft)
            except NotImplementedError:
                pass
            else:
                self.assertAlmostEqual(nrj_versatile, nrj)

            try:
                E_kx, E_ky, E_kz = op.compute_1dspectra(energy_fft)
            except NotImplementedError:
                pass
            else:
                self.assertAlmostEqual(E_kx.sum()*op.deltakx,
                                       E_ky.sum()*op.deltaky)

            try:
                E_k = op.compute_3dspectrum(energy_fft)
            except NotImplementedError:
                pass
            else:
                self.assertAlmostEqual(nrja, E_k.sum()*op.deltak_spectra3d)

            try:
                E_kx_kyz, E_ky_kzx, E_kz_kxy = op.compute_spectra_2vars(
                    energy_fft)
            except NotImplementedError:
                pass
            else:
                self.assertAlmostEqual(E_kx_kyz.sum()*op.deltakx,
                                       E_ky_kzx.sum()*op.deltaky)
                self.assertAlmostEqual(E_kz_kxy.sum()*op.deltakz, nrja)

            op.produce_str_describing_grid()
            op.produce_str_describing_oper()
            op.produce_long_str_describing_oper()
            op.create_arrayX(value=None, shape='loc')
            op.create_arrayX(value=None, shape='seq')
            op.create_arrayX(value=0.)

            op.project_perpk3d(afft, afft, afft)
            op.divfft_from_vecfft(afft, afft, afft)

            # depreciated...
            # op.vgradv_from_v(a, a, a)
            # op.vgradv_from_v2(a, a, a)
            # op.div_vv_fft_from_v(a, a, a)
            # op.div_vb_fft_from_vb(a, a, a, a)

            X, Y, Z = op.get_XYZ_loc()
            self.assertEqual(X.shape, op.shapeX_loc)
            self.assertEqual(Y.shape, op.shapeX_loc)
            self.assertEqual(Z.shape, op.shapeX_loc)

            if hasattr(op, 'gather_Xspace'):
                X = np.ascontiguousarray(X)
                Y = np.ascontiguousarray(Y)
                Z = np.ascontiguousarray(Z)
                X_seq = op.gather_Xspace(X)
                Y_seq = op.gather_Xspace(Y)
                Z_seq = op.gather_Xspace(Z)
            else:
                X_seq = X
                Y_seq = Y
                Z_seq = Z

            self.assertTrue(np.allclose(X_seq[0, 0, :], op.x_seq))
            self.assertTrue(np.allclose(Y_seq[0, :, 0], op.y_seq))
            self.assertTrue(np.allclose(Z_seq[:, 0, 0], op.z_seq))

        tests[key] = test

    return tests


class Tests3D(unittest.TestCase):
    pass


def complete_class(name, cls):

    tests = make_testop_functions(name, cls)

    for key, test in tests.items():
        setattr(Tests3D, 'test_operator3d_{}_{}'.format(name, key), test)


if nb_proc == 1:
    if nb_proc == 1 and len(classes_seq) == 0:
        raise Exception(
            'ImportError for all sequential classes. Nothing is working!')

    for name, cls in classes_seq.items():
        complete_class(name, cls)

if nb_proc > 1:
    if len(classes_mpi) == 0:
        raise Exception(
            'ImportError for all mpi classes. Nothing is working in mpi!')

    for name, cls in classes_mpi.items():
        complete_class(name, cls)


if __name__ == '__main__':
    unittest.main()
