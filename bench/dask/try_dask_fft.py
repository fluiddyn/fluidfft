import unittest
import numpy as np
import dask.array as da

from fluidfft.fft2d.operators import OperatorsPseudoSpectral2D
from fluidfft.fft2d.with_dask import FFTclass
from fluidfft.fft2d.test_2d import complete_class


def get_oper(fft):
    nx = ny = 1024
    lx = ly = 2 * np.pi
    return OperatorsPseudoSpectral2D(nx, ny, lx, ly, fft=fft)


class TestDask(unittest.TestCase):
    pass


complete_class("with_dask", FFTclass, TestDask)


if __name__ == "__main__":
    oper = get_oper("fft2d.with_dask")
    oper.opfft.run_tests()
    print("Basic tests pass for dask")

    # oper.opfft.run_benchs()
    # oper = get_oper("fft2d.with_pyfftw")
    # oper.opfft.run_benchs()

    unittest.main()
