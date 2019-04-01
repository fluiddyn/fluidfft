import unittest
from math import pi

from fluidfft.fft2d.operators import OperatorsPseudoSpectral2D
from fluidfft.fft2d.with_dask import FFTclass
from fluidfft.fft2d.test_2d import complete_class


def get_oper(fft):
    nx = ny = 128
    lx = ly = 2 * pi
    return OperatorsPseudoSpectral2D(nx, ny, lx, ly, fft=fft)


class TestDask(unittest.TestCase):
    pass


complete_class("with_dask", FFTclass, TestDask)


if __name__ == "__main__":
    oper = get_oper("fft2d.with_dask")
    oper_pyfftw = get_oper("fft2d.with_pyfftw")

    oper.opfft.run_tests()
    print("Basic tests pass for dask")

    print("Testing _numpy_api property... expecting dask")
    np = oper._numpy_api
    print(type(np.ones(2)))
    print("Testing _numpy_api property... expecting numpy")
    np = oper_pyfftw._numpy_api
    print(type(np.ones(2)))

    print("Benchmarking...")
    oper.opfft.run_benchs()
    oper_pyfftw.opfft.run_benchs()

    unittest.main()
