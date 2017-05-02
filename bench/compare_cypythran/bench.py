"""
In ipython:

run bench.py

# then (copy past the 4 lines):

%timeit gradfft_from_fft_py(f_fft, KX, KY)         # python
%timeit gradfft_from_fft(f_fft, KX, KY)            # pythran
# cython with @cython.boundscheck(False) @cython.wraparound(False)
%timeit gradfft_from_fft_nocheck_cy(f_fft, KX, KY)
# cython without @cython.boundscheck(False) @cython.wraparound(False)
%timeit gradfft_from_fft_check_cy(f_fft, KX, KY)

which gives

10 loops, best of 3: 17 ms per loop
100 loops, best of 3: 14.8 ms per loop
100 loops, best of 3: 9.52 ms per loop
100 loops, best of 3: 15 ms per loop

With perf::

  make perf

which gives:

# python
python -m perf timeit -s 'from bench import gradfft_from_fft_py as g, f_fft, KX, KY' 'g(f_fft, KX, KY)'
.....................
Mean +- std dev: 17.1 ms +- 0.4 ms
# pythran
python -m perf timeit -s 'from bench import gradfft_from_fft as g, f_fft, KX, KY' 'g(f_fft, KX, KY)'
.....................
Mean +- std dev: 15.0 ms +- 0.7 ms
# cython with @cython.boundscheck(False) @cython.wraparound(False)
python -m perf timeit -s 'from bench import gradfft_from_fft_nocheck_cy as g, f_fft, KX, KY' 'g(f_fft, KX, KY)'
.....................
Mean +- std dev: 9.46 ms +- 0.37 ms
# cython without @cython.boundscheck(False) @cython.wraparound(False)
python -m perf timeit -s 'from bench import gradfft_from_fft_check_cy as g, f_fft, KX, KY' 'g(f_fft, KX, KY)'
.....................
Mean +- std dev: 15.0 ms +- 0.8 ms

"""
from runpy import run_path

import numpy as np

import grad_pythran
from grad_pythran import gradfft_from_fft

from grad_cy import (
    gradfft_from_fft_nocheck as gradfft_from_fft_nocheck_cy,
    gradfft_from_fft_check as gradfft_from_fft_check_cy)

d = run_path('grad_pythran.py')
gradfft_from_fft_py = d['gradfft_from_fft']

assert hasattr(grad_pythran, '__pythran__')

n = 1000
shape = n, n

f_fft = np.ones(shape, dtype=np.complex128)

KX = np.ones(shape, dtype=np.float64)
KY = np.ones(shape, dtype=np.float64)

if __name__ == '__main__':

    gradfft_from_fft_py(f_fft, KX, KY)
    gradfft_from_fft(f_fft, KX, KY)
    gradfft_from_fft_nocheck_cy(f_fft, KX, KY)
    gradfft_from_fft_check_cy(f_fft, KX, KY)
