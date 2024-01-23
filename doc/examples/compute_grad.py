import numpy as np
from fluidfft.fft2d.operators import OperatorsPseudoSpectral2D

nx = ny = 100
lx = ly = 2 * np.pi

oper = OperatorsPseudoSpectral2D(nx, ny, lx, ly, fft="fft2d.with_fftw2d")

u = np.sin(oper.XX + oper.YY)
u_fft = oper.fft(u)
px_u_fft, py_u_fft = oper.gradfft_from_fft(u_fft)
px_u = oper.ifft(px_u_fft)
py_u = oper.ifft(py_u_fft)
grad_u = (px_u, py_u)
