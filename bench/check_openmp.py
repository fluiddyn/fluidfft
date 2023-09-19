# OMP_NUM_THREADS=1 python check_openmp.py
# OMP_NUM_THREADS=$(nproc) python check_openmp.py

import timeit

import numpy as np

np.show_config()


setup = (
'import numpy as np',
'from fluidfft.fft2d.operators import OperatorsPseudoSpectral2D'
'shape = (1024,)*2',
'oper = OperatorsPseudoSpectral2D(*shape, 1, 1)'
'divfft_from_vecfft = oper.divfft_from_vecfft',
'a_fft = np.ones(shape, dtype=np.complex128)',
'b_fft = np.ones(shape, dtype=np.complex128)',
'kx = np.ones(shape, dtype=np.float64)',
'ky = np.ones(shape, dtype=np.float64)',
)
setup = ';'.join(setup)
cmd = 'div_fft = divfft_from_vecfft(a_fft, b_fft, kx, ky)'

nb_iter = 2000

t = timeit.Timer(cmd, setup=setup)
print('divfft_from_vecfft: {} sec / iteration for {} iterations'.format(
    t.timeit(nb_iter) / nb_iter, nb_iter))
