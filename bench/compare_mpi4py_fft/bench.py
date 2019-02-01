import numpy as np
from mpi4py import MPI
from mpi4py_fft.mpifft import PFFT, Function
from time import time


N = np.array([128, 128, 128], dtype=int)
fft = PFFT(MPI.COMM_WORLD, N, axes=(0, 1, 2), dtype=np.float)
u = Function(fft, False)
u[:] = np.random.random(u.shape).astype(u.dtype)

nb_iter = 20
tstart = time()
for _ in range(nb_iter):
    u_hat = fft.forward(u)
tend = time()

print(f"Avg time for fft =", (tend-tstart)/nb_iter)

uj = np.zeros_like(u)
tstart = time()
for _ in range(nb_iter):
    uj = fft.backward(u_hat, uj)
tend = time()

print(f"Avg time for ifft =", (tend-tstart)/nb_iter)
assert np.allclose(uj, u)
print(MPI.COMM_WORLD.Get_rank(), u.shape, u.dtype, u_hat.dtype)
