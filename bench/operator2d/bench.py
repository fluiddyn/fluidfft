
from time import time

import numpy as np

from fluiddyn.util.paramcontainer import ParamContainer
from fluidsim.operators.operators import OperatorsPseudoSpectral2D

import openmp, default, native


def create_oper_cython(shape):
    params = ParamContainer(tag='params')

    params._set_attrib('ONLY_COARSE_OPER', False)

    OperatorsPseudoSpectral2D._complete_params_with_default(params)

    params.oper.nx = shape[1]
    params.oper.ny = shape[0]

    return OperatorsPseudoSpectral2D(params=params)

shape = [512*2]*2

oper = create_oper_cython(shape)


modules = {'omp': openmp, 'default': default, 'native': native}

f_fft = oper.constant_arrayK(0)
KX = f_fft.real.astype(np.float64)

keys = list(modules.keys())
keys.sort()

for k in keys:
    mod = modules[k]
    print('bench', k)

    nb_times = 1000
    t_start = time()
    for i in range(nb_times):
        # mod.util.divfft_from_vecfft(f_fft, f_fft, KX, KX)
        mod.util.gradfft_from_fft(f_fft, KX, KX)
        # mod.util.myfunc(a)

    duration = time() - t_start

    print('duration = {:3.2f} ms ({})'.format(duration/nb_times*1000, k))

print('bench cython')

t_start = time()
for i in range(1000):
    # oper.divfft_from_vecfft(f_fft, f_fft)
    oper.gradfft_from_fft(f_fft)

duration = time() - t_start

print('duration = {:3.2f} ms (cython)'.format(duration/nb_times*1000))
