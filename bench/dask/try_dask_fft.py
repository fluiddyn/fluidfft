import unittest
from math import pi

from fluidfft.fft2d.operators import OperatorsPseudoSpectral2D
from fluidfft.fft2d.with_dask import FFTclass
from fluidfft.fft2d.test_2d import complete_class


def get_oper(fft):
    nx = ny = 16
    lx = ly = 2 * pi
    return OperatorsPseudoSpectral2D(nx, ny, lx, ly, fft=fft)


class TestDask(unittest.TestCase):
    pass


complete_class("with_dask", FFTclass, TestDask)


if __name__ == "__main__":
    oper_dask = get_oper("fft2d.with_dask")
    oper_pyfftw = get_oper("fft2d.with_pyfftw")
    oper_fftw = get_oper("fft2d.with_fftw2d")

    for op in (oper_dask, oper_pyfftw, oper_fftw):
        print('=' * 50)
        print(op.type_fft)
        print('=' * 50)
        op.opfft.run_tests()
        print("Basic tests pass")

        print("Testing _numpy_api property...")
        np = op._numpy_api
        print(type(np.ones(2)))

        print("Create array methods")
        print(op.create_arrayX(value=2.0).dtype)
        print(op.create_arrayK(value=2.0).dtype)
        print(op.create_arrayX_random().dtype)
        print(op.create_arrayK_random().dtype)

        print("Benchmarking...")
        op.opfft.run_benchs()

    unittest.main()
