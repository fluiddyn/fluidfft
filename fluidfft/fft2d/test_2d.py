import unittest
from math import pi
import traceback

import numpy as np

from fluiddyn.util import mpi

from fluidfft.fft2d import get_classes_seq, get_classes_mpi
from fluidfft.fft2d.operators import OperatorsPseudoSpectral2D

try:
    import fluidfft.fft2d.with_fftw2d
except ImportError:
    # If this one does not work it is a bad sign so we want to know what happened.
    traceback.print_exc()


n = 24

rank = mpi.rank
nb_proc = mpi.nb_proc

classes_seq = get_classes_seq()
classes_seq = {name: cls for name, cls in classes_seq.items() if cls is not None}

if not classes_seq:
    raise ImportError("Not sequential 2d classes working!")

if nb_proc > 1:
    classes_mpi = get_classes_mpi()
    classes_mpi = {
        name: cls for name, cls in classes_mpi.items() if cls is not None
    }


def make_test_function(cls):
    def test(self):
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

            nrja_mean_global = op.mean_global(0.5 * a ** 2)
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


class Tests2D(unittest.TestCase):
    pass


def complete_class(name, cls, Tests2D=Tests2D):

    if cls is not None:
        setattr(Tests2D, "test_{}".format(name), make_test_function(cls))

    tests = make_testop_functions(name, cls)

    for key, test in tests.items():
        setattr(Tests2D, "test_operator2d_{}_{}".format(name, key), test)


if rank == 0:
    if nb_proc == 1 and len(classes_seq) == 0:
        raise Exception(
            "ImportError for all sequential classes. Nothing is working!"
        )

    for name, cls in classes_seq.items():
        complete_class(name, cls)


if nb_proc > 1:
    if len(classes_mpi) == 0:
        raise Exception(
            "ImportError for all mpi classes. Nothing is working in mpi!"
        )

    for name, cls in classes_mpi.items():
        complete_class(name, cls)


complete_class("None", None)


if __name__ == "__main__":
    unittest.main()
