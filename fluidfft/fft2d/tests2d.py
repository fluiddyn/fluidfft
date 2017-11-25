
from __future__ import print_function, division

import unittest
from math import pi

import numpy as np

from fluiddyn.util import mpi

from fluidfft.fft2d import get_classes_seq, get_classes_mpi
from fluidfft.fft2d.operators import OperatorsPseudoSpectral2D

# to check that at least this class can be imported
import fluidfft.fft2d.with_fftw1d

n = 32

rank = mpi.rank
nb_proc = mpi.nb_proc

classes_seq = get_classes_seq()
classes_seq = {name: cls for name, cls in classes_seq.items()
               if cls is not None}

if nb_proc > 1:
    classes_mpi = get_classes_mpi()
    classes_mpi = {name: cls for name, cls in classes_mpi.items()
                   if cls is not None}


def make_test_function(cls):

    def test(self):
        o = cls(n, n)
        o.run_tests()

        a = np.random.random(o.get_local_size_X()).reshape(
            o.get_shapeX_loc())
        afft = o.fft(a)
        a = o.ifft(afft)
        afft = o.fft(a)

        EX = o.compute_energy_from_X(a)
        EK = o.compute_energy_from_K(afft)

        self.assertTrue(EX == EK and EX != 0.)

        k0, k1 = o.get_k_adim_loc()

        if o.get_shapeK_loc() != (k0.size, k1.size):
            print(o.get_shapeK_loc(), k0.size, k1.size)
            raise Exception('o.get_shapeK_loc() != (k0.size, k1.size)')
    return test


def make_testop_functions(name, cls):

    tests = {}
    shapes = {'even': (8, 4)}
    if nb_proc == 1:
        shapes['odd'] = (11, 13)

    for key, (n0, n1) in shapes.items():

        def test(self, n0=n0, n1=n1):
            op = OperatorsPseudoSpectral2D(n0, n1, 3*pi, 1., fft=cls)
            a = np.random.random(op.opfft.get_local_size_X()).reshape(
                op.opfft.get_shapeX_loc())
            afft = op.fft(a)
            a = op.ifft(afft)
            afft = op.fft(a)

            nrja = op.compute_energy_from_X(a)
            nrjafft = op.compute_energy_from_K(afft)
            self.assertEqual(nrja, nrjafft)

            # print('energy', nrja)
            energy_fft = 0.5 * abs(afft)**2

            E_kx, E_ky = op.compute_1dspectra(energy_fft)

            self.assertAlmostEqual(E_kx.sum()*op.deltakx,
                                   E_ky.sum()*op.deltaky)

            E_kh = op.compute_2dspectrum(energy_fft)

            self.assertAlmostEqual(nrja, E_kh.sum()*op.deltakh)

            op.sum_wavenumbers(energy_fft)
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

        tests[key] = test

    return tests


class Tests2D(unittest.TestCase):
    pass


def complete_class(name, cls):

    setattr(Tests2D, 'test_{}'.format(name), make_test_function(cls))

    tests = make_testop_functions(name, cls)

    for key, test in tests.items():
        setattr(Tests2D, 'test_operator2d_{}_{}'.format(name, key), test)


if rank == 0:
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
