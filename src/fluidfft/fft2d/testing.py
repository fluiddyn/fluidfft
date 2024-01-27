from math import pi

import numpy as np

from fluiddyn.util import mpi

from fluidfft import import_fft_class
from fluidfft.fft2d.operators import OperatorsPseudoSpectral2D


rank = mpi.rank
nb_proc = mpi.nb_proc


def make_test_function(cls):
    def test(self):
        n = 24
        o = cls(n, 2 * n)
        o.run_tests()
        a = np.random.rand(*o.get_shapeX_loc())
        afft = o.fft(a)
        a = o.ifft(afft)
        afft = o.fft(a)

        EX = o.compute_energy_from_X(a)
        EK = o.compute_energy_from_K(afft)

        self.assertTrue(EX != 0.0)
        self.assertAlmostEqual(EX, EK)

        k0, k1 = o.get_k_adim_loc()

        if o.get_shapeK_loc() != (k0.size, k1.size):
            print(o.get_shapeK_loc(), k0.size, k1.size)
            raise Exception("o.get_shapeK_loc() != (k0.size, k1.size)")

    return test


def make_testop_functions(name, cls):
    tests = {}
    shapes = {"even": (24, 16)}
    if nb_proc == 1:
        shapes["odd"] = (11, 13)

    for key, (n0, n1) in shapes.items():

        def test(self, n0=n0, n1=n1):
            op = OperatorsPseudoSpectral2D(n0, n1, 3 * pi, 1.0, fft=cls)
            op.create_arrayX(value=1.0, shape="seq")
            op.create_arrayK(value=1.0, shape="seq")

            a = op.create_arrayX_random(max_val=2)
            a0 = a.copy()
            afft = op.fft(a)
            self.assertTrue(np.allclose(a, a0))
            afft0 = afft.copy()
            a = op.ifft(afft)
            self.assertTrue(np.allclose(afft, afft0))
            afft = op.fft(a)

            # MPI Scatter-Gather tests
            if not op.is_sequential:
                if mpi.rank == 0:
                    arr_seq = op.create_arrayX_random(shape="seq")
                else:
                    arr_seq = None

                arr_loc = op.scatter_Xspace(arr_seq)
                arr_loc *= 2

                # Note: Specifying root to avoid Allgather
                arr_seq2 = op.gather_Xspace(arr_loc, root=0)
                if mpi.rank == 0:
                    np.testing.assert_array_equal(arr_seq * 2, arr_seq2)

            nrja = op.compute_energy_from_X(a)
            nrjafft = op.compute_energy_from_K(afft)
            self.assertAlmostEqual(nrja, nrjafft)

            nrja_mean_global = op.mean_global(0.5 * a**2)
            self.assertAlmostEqual(nrja, nrja_mean_global)

            # print('energy', nrja)
            energy_fft = 0.5 * abs(afft) ** 2

            try:
                nrj_versatile = op.sum_wavenumbers_versatile(energy_fft)
            except NotImplementedError:
                pass
            else:
                self.assertAlmostEqual(nrj_versatile, nrja)

            E_kx, E_ky = op.compute_1dspectra(energy_fft)
            self.assertAlmostEqual(
                E_kx.sum() * op.deltakx, E_ky.sum() * op.deltaky
            )

            E_kh = op.compute_2dspectrum(energy_fft)
            self.assertAlmostEqual(nrja, E_kh.sum() * op.deltak)

            try:
                E_ky_kx = op.compute_spectrum_kykx(energy_fft)
                E_ky_kx_uf = op.compute_spectrum_kykx(energy_fft, folded=False)
            except NotImplementedError:
                pass
            else:
                self.assertAlmostEqual(
                    E_ky_kx.sum() * op.deltakx * op.deltaky, nrja
                )
                self.assertAlmostEqual(
                    E_ky_kx_uf.sum() * op.deltakx * op.deltaky, nrja
                )

            nrj_sw = op.sum_wavenumbers(energy_fft)
            self.assertAlmostEqual(nrja, nrj_sw)

            op.produce_str_describing_grid()
            op.produce_str_describing_oper()
            op.produce_long_str_describing_oper()

            op.projection_perp(afft, afft)

            rot_fft = afft
            vecx_fft, vecy_fft = op.vecfft_from_rotfft(rot_fft)
            rot_fft = op.rotfft_from_vecfft(vecx_fft, vecy_fft)
            vecx_fft, vecy_fft = op.vecfft_from_rotfft(rot_fft)
            rot_fft_back = op.rotfft_from_vecfft(vecx_fft, vecy_fft)
            self.assertTrue(np.allclose(rot_fft, rot_fft_back))

            div_fft = afft
            vecx_fft, vecy_fft = op.vecfft_from_divfft(div_fft)
            div_fft = op.divfft_from_vecfft(vecx_fft, vecy_fft)
            vecx_fft, vecy_fft = op.vecfft_from_divfft(div_fft)
            div_fft_back = op.divfft_from_vecfft(vecx_fft, vecy_fft)
            self.assertTrue(np.allclose(div_fft, div_fft_back))

            op.gradfft_from_fft(afft)
            op.dealiasing_variable(afft)

            op.create_arrayK_random(min_val=-1, max_val=1)

        tests[key] = test

    return tests


def complete_test_class_2d(method, test_class, cls=None):
    if cls is None:
        cls = import_fft_class(method)

    short_name = method.split(".")[1]

    setattr(test_class, f"test_{short_name}", make_test_function(cls))

    tests = make_testop_functions(method, cls)

    for key, test in tests.items():
        setattr(test_class, f"test_operator2d_{short_name}_{key}", test)
