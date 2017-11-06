
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


def make_test_function(cls, sequential=False):

    def test(self):
        if sequential and rank > 0:
            return
        o = cls(n, n, n)
        o.run_tests()

    return test


def make_testop_functions(name, cls):

    tests = {}
    shapes = {'even': (8, 8, 8)}
    if nb_proc == 1:
        shapes['odd'] = (5, 3, 3)

    for key, (n0, n1, n2) in shapes.items():

        def test(self, n0=n0, n1=n1, n2=n2):

            op = OperatorsPseudoSpectral3D(n0, n1, n2,
                                           3*pi, 1*pi, 2*pi, fft=cls)
            a = np.random.random(
                op._op_fft.get_local_size_X()).reshape(
                    op._op_fft.get_shapeX_loc())
            afft = op.fft3d(a)
            a = op.ifft3d(afft)
            afft = op.fft3d(a)

            nrja = op.compute_energy_from_X(a)
            nrjafft = op.compute_energy_from_K(afft)
            self.assertEqual(nrja, nrjafft)

            energy_fft = 0.5 * abs(afft)**2
            nrj = op.sum_wavenumbers(energy_fft)
            self.assertAlmostEqual(nrjafft, nrj)

            # not implemented...
            # E_kx, E_ky, E_kz = op.compute_1dspectra(energy_fft)
            # self.assertAlmostEqual(E_kx.sum()*op.deltakx,
            #                        E_ky.sum()*op.deltaky)

            # E_kh = op.compute_2dspectrum(energy_fft)
            # self.assertAlmostEqual(nrja, E_kh.sum()*op.deltakh)

        tests[key] = test

    return tests


class Tests3D(unittest.TestCase):
    pass


def complete_class(name, cls):

    # setattr(Tests3D, 'test_{}'.format(name), make_test_function(cls))

    # setattr(Tests3D, 'test_{}_seq'.format(name),
    #         make_test_function(cls, sequential=1))

    tests = make_testop_functions(name, cls)

    for key, test in tests.items():
        setattr(Tests3D, 'test_operator3d_{}_{}'.format(name, key), test)


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
