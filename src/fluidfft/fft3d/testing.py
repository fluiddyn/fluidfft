import numpy as np

import pytest

from fluiddyn.util import mpi

from fluidfft import import_fft_class
from .operators import OperatorsPseudoSpectral3D, vector_product

rank = mpi.rank
nb_proc = mpi.nb_proc


def complete_test_class_3d(
    method, test_class, cls=None, skip_if_import_error=True
):
    short_name = method.split(".")[1]
    if cls is None:
        cls = import_fft_class(method, raise_import_error=skip_if_import_error)

    tests = make_testop_functions(cls)
    if cls is None:
        tests = {
            key: pytest.mark.skip(reason="Class not importable")(function)
            for key, function in tests.items()
        }
    for key, test in tests.items():
        setattr(test_class, f"test_operator3d_{short_name}_{key}", test)


def make_testop_functions(cls):
    tests = {}
    shapes = {"even": (4, 8, 12)}
    if nb_proc == 1:
        shapes["odd"] = (5, 3, 7)

    for key, (n0, n1, n2) in shapes.items():

        def test(self, n0=n0, n1=n1, n2=n2):
            try:
                op = OperatorsPseudoSpectral3D(n2, n1, n0, 12, 8, 4, fft=cls)
            except ValueError:
                print(
                    "ValueError while instantiating OperatorsPseudoSpectral3D"
                    " for {}".format(cls)
                )
                return

            op_fft = op._op_fft

            op_fft.run_tests()

            a = np.random.random(op_fft.get_local_size_X()).reshape(
                op_fft.get_shapeX_loc()
            )
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

            energy_fft = 0.5 * abs(afft) ** 2
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
                self.assertAlmostEqual(nrj, E_kx.sum() * op.deltakx)
                self.assertAlmostEqual(nrj, E_ky.sum() * op.deltaky)
                self.assertAlmostEqual(nrj, E_kz.sum() * op.deltakz)

                self.assertEqual(E_kx.shape[0], op.nkx_spectra)
                self.assertEqual(E_ky.shape[0], op.nky_spectra)
                self.assertEqual(E_kz.shape[0], op.nkz_spectra)

            E_k = op.compute_3dspectrum(energy_fft)
            self.assertAlmostEqual(nrja, E_k.sum() * op.deltak_spectra3d)

            E_kz_kh = op.compute_spectrum_kzkh(energy_fft)
            self.assertAlmostEqual(nrja, E_kz_kh.sum() * op.deltakh * op.deltakz)

            try:
                E_kx_kyz, E_ky_kzx, E_kz_kxy = op.compute_spectra_2vars(
                    energy_fft
                )
            except NotImplementedError:
                pass
            else:
                self.assertAlmostEqual(
                    E_kx_kyz.sum() * op.deltakx, E_ky_kzx.sum() * op.deltaky
                )
                self.assertAlmostEqual(E_kz_kxy.sum() * op.deltakz, nrja)

            op.produce_str_describing_grid()
            op.produce_str_describing_oper()
            op.produce_long_str_describing_oper()
            op.create_arrayX(value=None, shape="loc")
            op.create_arrayX(value=None, shape="seq")
            op.create_arrayX(value=0.0)
            op.create_arrayK(value=1.0)
            op.create_arrayX_random(max_val=2)
            op.create_arrayK_random(min_val=-1, max_val=1, shape="seq")

            op.project_perpk3d(afft, afft, afft)
            op.divfft_from_vecfft(afft, afft, afft)
            op.rotfft_from_vecfft(afft, afft, afft)
            op.rotfft_from_vecfft_outin(afft, afft, afft, afft, afft, afft)
            op.rotzfft_from_vxvyfft(afft, afft)

            # depreciated...
            # op.vgradv_from_v(a, a, a)
            # op.vgradv_from_v2(a, a, a)
            # op.div_vv_fft_from_v(a, a, a)
            op.div_vb_fft_from_vb(a, a, a, a)
            vector_product(a, a, a, a, a, a)

            X, Y, Z = op.get_XYZ_loc()
            self.assertEqual(X.shape, op.shapeX_loc)
            self.assertEqual(Y.shape, op.shapeX_loc)
            self.assertEqual(Z.shape, op.shapeX_loc)

            X = np.ascontiguousarray(X)
            Y = np.ascontiguousarray(Y)
            Z = np.ascontiguousarray(Z)
            root = 0
            X_seq = op.gather_Xspace(X, root=root)
            Y_seq = op.gather_Xspace(Y, root=root)
            Z_seq = op.gather_Xspace(Z, root=root)

            if rank == root:
                self.assertTrue(np.allclose(X_seq[0, 0, :], op.x_seq))
                self.assertTrue(np.allclose(Y_seq[0, :, 0], op.y_seq))
                self.assertTrue(np.allclose(Z_seq[:, 0, 0], op.z_seq))

            X_scatter = op.scatter_Xspace(X_seq, root=root)
            Y_scatter = op.scatter_Xspace(Y_seq, root=root)
            Z_scatter = op.scatter_Xspace(Z_seq, root=root)

            self.assertTrue(np.allclose(X, X_scatter))
            self.assertTrue(np.allclose(Y, Y_scatter))
            self.assertTrue(np.allclose(Z, Z_scatter))

        tests[key] = test

    return tests
