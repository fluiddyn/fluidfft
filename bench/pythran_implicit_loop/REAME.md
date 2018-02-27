# Benchmarks comparing pythran extensions compiled with different flags

# Result on my machine at LEGI (Intel(R) Xeon(R) CPU E5-1603 v3 @ 2.80GHz, 4 cores).

## numpy
python -m perf timeit -s 'import numpy as np; a = np.ones([1000, 1000]); from mymod import myfunc' 'myfunc(a)'
.....................
Mean +- std dev: 47.6 ms +- 0.2 ms

## default
python -m perf timeit -s 'import numpy as np; a = np.ones([1000, 1000]); from mymod_default import myfunc' 'myfunc(a)'
.....................
Mean +- std dev: 5.15 ms +- 0.01 ms

## native
python -m perf timeit -s 'import numpy as np; a = np.ones([1000, 1000]); from mymod_native import myfunc' 'myfunc(a)'
.....................
Mean +- std dev: 5.15 ms +- 0.01 ms

## native_openmp
python -m perf timeit -s 'import numpy as np; a = np.ones([1000, 1000]); from mymod_native_openmp import myfunc' 'myfunc(a)'
.....................
Mean +- std dev: 1.41 ms +- 0.11 ms

## openmp
python -m perf timeit -s 'import numpy as np; a = np.ones([1000, 1000]); from mymod_openmp import myfunc' 'myfunc(a)'
.....................
Mean +- std dev: 1.40 ms +- 0.06 ms

## simd
python -m perf timeit -s 'import numpy as np; a = np.ones([1000, 1000]); from mymod_simd import myfunc' 'myfunc(a)'
.....................
Mean +- std dev: 5.16 ms +- 0.01 ms


fopenmp is really efficient (speed up of 3.65 on 4 cores...). In contrast, we
see no effect of the flags -march=native or -DUSE_BOOST_SIMD. Why?
