
import numpy as np

from fluidfft.fft2d.util_pythran import divfft_from_vecfft

shape = (1024,)*2

a_fft = np.ones(shape, dtype=np.complex128)
b_fft = np.ones(shape, dtype=np.complex128)

kx = np.ones(shape, dtype=np.float64)
ky = np.ones(shape, dtype=np.float64)

for i in range(2000):
    div_fft = divfft_from_vecfft(a_fft, b_fft, kx, ky)

