
from __future__ import print_function, division

import unittest

import numpy as np

from math import pi

from fluiddyn.util import mpi

from fluidfft.fft2d import get_classes_seq, get_classes_mpi

n = 128

rank = mpi.rank
nb_proc = mpi.nb_proc

classes = get_classes_seq()

if nb_proc == 1:
    from fluidfft.fft2d.operator import OperatorPseudoSpectral2D
if nb_proc > 1:
    classes.update(get_classes_mpi())


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


def make_testop_function(name):

    def test(self):
        op = OperatorPseudoSpectral2D(25, 15, 3*pi, 1*pi)
        a = np.random.random(op._opfft.get_local_size_X()).reshape(
            op._opfft.get_shapeX_loc())
        afft = op.fft(a)
        a = op.ifft(afft)
        afft = op.fft(a)

        nrja = op.compute_energy_from_X(a)
        nrjafft = op.compute_energy_from_K(afft)
        self.assertEqual(nrja, nrjafft)

        nrjspa = (a**2).mean()/2

        energy_fft = 0.5 * abs(afft)**2

        E_kx, E_ky = op.compute_1dspectra(energy_fft)

        self.assertAlmostEqual(E_kx.sum()*op.deltakx, E_ky.sum()*op.deltaky)

        E_kh = op.compute_2dspectrum(energy_fft)

        self.assertAlmostEqual(nrjspa, E_kh.sum()*op.deltakh)
    return test


class Tests2D(unittest.TestCase):
    pass


for name, cls in classes.items():
    if cls is None:
        continue

    setattr(Tests2D, 'test_{0}'.format(name), make_test_function(cls))

if nb_proc == 1:
    setattr(Tests2D, 'test_operator2d', make_testop_function(name))


if __name__ == '__main__':
    unittest.main()
