"""Example

https://bitbucket.org/mpi4py/mpi4py-fft/src/master/examples/transforms.py
Copyright (c) 2017, Lisandro Dalcin and Mikael Mortensen. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    Redistributions of source code must retain the above copyright notice, this
    list of conditions and the following disclaimer.  Redistributions in binary
    form must reproduce the above copyright notice, this list of conditions and
    the following disclaimer in the documentation and/or other materials
    provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import functools
import numpy as np
from mpi4py import MPI
from mpi4py_fft.mpifft import PFFT, Function
from mpi4py_fft.fftw import dctn, idctn

# Set global size of the computational box
N = np.array([18, 18, 18], dtype=int)

dct = functools.partial(dctn, type=3)
idct = functools.partial(idctn, type=3)

transforms = {(1, 2): (dct, idct)}

fft = PFFT(MPI.COMM_WORLD, N, axes=None, collapse=True, slab=True, transforms=transforms)
pfft = PFFT(MPI.COMM_WORLD, N, axes=((0,), (1, 2)), slab=True, padding=[1.5, 1.0, 1.0], transforms=transforms)

assert fft.axes == pfft.axes

u = Function(fft, False)
u[:] = np.random.random(u.shape).astype(u.dtype)

u_hat = Function(fft)
u_hat = fft.forward(u, u_hat)
uj = np.zeros_like(u)
uj = fft.backward(u_hat, uj)
assert np.allclose(uj, u)

u_padded = Function(pfft, False)
uc = u_hat.copy()
u_padded = pfft.backward(u_hat, u_padded)
u_hat = pfft.forward(u_padded, u_hat)
assert np.allclose(u_hat, uc)

#cfft = PFFT(MPI.COMM_WORLD, N, dtype=complex, padding=[1.5, 1.5, 1.5])
cfft = PFFT(MPI.COMM_WORLD, N, dtype=complex)

uc = np.random.random(cfft.backward.input_array.shape).astype(complex)
u2 = cfft.backward(uc)
u3 = uc.copy()
u3 = cfft.forward(u2, u3)

assert np.allclose(uc, u3)


