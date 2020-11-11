# Benchmarks comparing pythran extensions compiled with different flags

# Result on Pierre's machine at LEGI (Intel(R) Xeon(R) CPU E5-1603 v3 @ 2.80GHz, 4 cores).

## numpy
python -m pyperf timeit -s 'import numpy as np; a = np.ones([1000, 1000]); from mymod import myfunc' 'myfunc(a)'
.....................
Mean +- std dev: 47.6 ms +- 0.2 ms

## default
python -m pyperf timeit -s 'import numpy as np; a = np.ones([1000, 1000]); from mymod_default import myfunc' 'myfunc(a)'
.....................
Mean +- std dev: 5.15 ms +- 0.01 ms

## native
python -m pyperf timeit -s 'import numpy as np; a = np.ones([1000, 1000]); from mymod_native import myfunc' 'myfunc(a)'
.....................
Mean +- std dev: 5.15 ms +- 0.01 ms

## native_openmp
python -m pyperf timeit -s 'import numpy as np; a = np.ones([1000, 1000]); from mymod_native_openmp import myfunc' 'myfunc(a)'
.....................
Mean +- std dev: 1.41 ms +- 0.11 ms

## openmp
python -m pyperf timeit -s 'import numpy as np; a = np.ones([1000, 1000]); from mymod_openmp import myfunc' 'myfunc(a)'
.....................
Mean +- std dev: 1.40 ms +- 0.06 ms

## simd
python -m pyperf timeit -s 'import numpy as np; a = np.ones([1000, 1000]); from mymod_simd import myfunc' 'myfunc(a)'
.....................
Mean +- std dev: 5.16 ms +- 0.01 ms


fopenmp is really efficient (speed up of 3.65 on 4 cores...). In contrast, we
see no effect of the flags -march=native or -DUSE_BOOST_SIMD. Why?


# Update (2018-10-01)
The following are the results, in Ashwin's laptop after a `sudo python -m pyperf system tune`.
(Intel(R) Core(TM) i7-5500U CPU @ 2.40GHz, 2 cores).

### branch master (6f2a8f6)

```sh
# with gcc 8.2.1 (with an empty `~/.pythranrc`)
## default 2d (no loop)
Mean +- std dev: 18.3 ms +- 0.3 ms
# default 2d (explicit loops)
Mean +- std dev: 11.4 ms +- 0.0 ms
# simd 2d (no loop)
Mean +- std dev: 18.7 ms +- 0.2 ms
# simd 2d (explicit loops)
Mean +- std dev: 11.4 ms +- 0.1 ms

# with gcc 8.2.1 (with `complex_hook=True` inside `~/.pythranrc`)
## default 2d (no loop)
Mean +- std dev: 5.48 ms +- 0.49 ms
# default 2d (explicit loops)
Mean +- std dev: 4.50 ms +- 0.11 ms
# simd 2d (no loop)
Mean +- std dev: 5.20 ms +- 0.03 ms
# simd 2d (explicit loops)
Mean +- std dev: 4.33 ms +- 0.01 ms
```

### branch fix/complex-scalar-broadcast-perf (9d18ec6)

```sh
# with gcc 8.2.1 (with an empty `~/.pythranrc`)
$ make perf2d
## default 2d (no loop)
Mean +- std dev: 14.2 ms +- 0.2 ms
# default 2d (explicit loops)
Mean +- std dev: 10.9 ms +- 0.0 ms
# simd 2d (no loop)
Mean +- std dev: 14.2 ms +- 0.3 ms
# simd 2d (explicit loops)
Mean +- std dev: 10.9 ms +- 0.1 ms

# with gcc 8.2.1 (with `complex_hook=True` inside `~/.pythranrc`)
$ make perf2d
## default 2d (no loop)
Mean +- std dev: 6.07 ms +- 0.07 ms
# default 2d (explicit loops)
Mean +- std dev: 4.45 ms +- 0.14 ms
# simd 2d (no loop)
python -m pyperf timeit -s \
Mean +- std dev: 5.49 ms +- 0.14 ms
# simd 2d (explicit loops)
Mean +- std dev: 4.40 ms +- 0.15 ms

```
