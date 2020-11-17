import sys
import numpy as np
import pyperf
from mpi4py import MPI
from mpi4py_fft.mpifft import PFFT
from mpi4py_fft.distarray import newDistArray


def init2d(N=1024, slab=True):
    return PFFT(MPI.COMM_WORLD, (N, N), slab=slab, axes=(0, 1), dtype=np.float)


def init3d(N=128, slab=True):
    return PFFT(
        MPI.COMM_WORLD, (N, N, N), slab=slab, axes=(0, 1, 2), dtype=np.float
    )


def create_arrayX(o):
    u = newDistArray(o, False)
    u[:] = np.random.random(u.shape).astype(u.dtype)
    return u


def create_arrayK(o):
    u_hat = newDistArray(o, True)
    u_hat[:] = np.random.random(u_hat.shape).astype(u_hat.dtype)
    return u_hat


def fft(o, u, u_hat):
    o.forward(u, u_hat)


def ifft(o, u, u_hat):
    o.backward(u_hat, u)
