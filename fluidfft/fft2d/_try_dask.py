import numpy as np
from fluidfft.fft2d.operators import OperatorsPseudoSpectral2D
import dask.array as da


def get_oper(fft):
    nx = ny = 1024
    lx = ly = 2 * np.pi
    return OperatorsPseudoSpectral2D(nx, ny, lx, ly, fft=fft)


oper = get_oper("fft2d.with_dask")
oper.opfft.run_tests()
oper.opfft.run_benchs()
print("Basic tests pass for dask")

oper = get_oper("fft2d.with_pyfftw")
oper.opfft.run_benchs()
