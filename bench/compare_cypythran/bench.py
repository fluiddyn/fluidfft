"""
In ipython
----------

Python 3.6.4 | packaged by conda-forge | (default, Dec 23 2017, 16:31:06) 
Type 'copyright', 'credits' or 'license' for more information
IPython 6.2.1 -- An enhanced Interactive Python. Type '?' for help.

In [1]: run bench.py

In [2]: %timeit grad_omp(f_fft, KX, KY)
9.15 ms ± 70 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

In [3]: %timeit grad_simd(f_fft, KX, KY)
8.91 ms ± 250 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

In [4]: %timeit grad_py(f_fft, KX, KY)
15.3 ms ± 54.2 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

In [5]: %timeit grad_pythran(f_fft, KX, KY)
8.75 ms ± 28.9 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

In [6]: %timeit grad_cy_nocheck(f_fft, KX, KY)
8.89 ms ± 257 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

With perf
---------

Compile::

  make

Launch benchmarking::

  make perf

which gives:

# python
python -m pyperf timeit -s \
  'from bench import grad_py as g, f_fft, KX, KY' 'g(f_fft, KX, KY)'
.....................
Mean +- std dev: 15.0 ms +- 0.3 ms
# cython with @cython.boundscheck(False) @cython.wraparound(False)
python -m pyperf timeit -s \
  'from bench import grad_cy_nocheck as g, f_fft, KX, KY' 'g(f_fft, KX, KY)'
.....................
Mean +- std dev: 8.76 ms +- 0.32 ms
# pythran
python -m pyperf timeit -s \
  'from bench import grad_pythran as g, f_fft, KX, KY' 'g(f_fft, KX, KY)'
.....................
Mean +- std dev: 8.69 ms +- 0.20 ms
# SIMD
python -m pyperf timeit -s \
  'from bench import grad_simd as g, f_fft, KX, KY' 'g(f_fft, KX, KY)'
.....................
Mean +- std dev: 8.91 ms +- 0.53 ms
# OpenMP
python -m pyperf timeit -s \
  'from bench import grad_omp as g, f_fft, KX, KY' 'g(f_fft, KX, KY)'
.....................
Mean +- std dev: 9.12 ms +- 0.33 ms

"""
from runpy import run_path

import numpy as np

import grad_pythran as mod_pythran

from grad_pythran import gradfft_from_fft as grad_pythran

from grad_simd import gradfft_from_fft as grad_simd
from grad_omp import gradfft_from_fft as grad_omp

from grad_cy import (
    gradfft_from_fft_nocheck as grad_cy_nocheck,
    gradfft_from_fft_check as grad_cy_check,
)

d = run_path("grad_pythran.py")
grad_py = d["gradfft_from_fft"]

assert hasattr(mod_pythran, "__pythran__")

n = 1000
shape = n, n

f_fft = np.ones(shape, dtype=np.complex128)

KX = np.ones(shape, dtype=np.float64)
KY = np.ones(shape, dtype=np.float64)

if __name__ == "__main__":

    grad_py(f_fft, KX, KY)
    grad_cy_nocheck(f_fft, KX, KY)
    grad_cy_check(f_fft, KX, KY)
    grad_pythran(f_fft, KX, KY)
    grad_simd(f_fft, KX, KY)
    grad_omp(f_fft, KX, KY)
