
name=mymod

setup=import numpy as np; a = np.ones([1000, 1000], dtype=np.complex128);
# setup=import numpy as np; a = np.ones([1000, 1000]);

compile:
	pythran -v $(name).py -o $(name)_default.so
	pythran -v $(name).py -march=native -o $(name)_native.so
	pythran -v $(name).py -march=native -fopenmp -o $(name)_native_omp.so
	pythran -v $(name).py -fopenmp -o $(name)_omp.so
	pythran -v $(name).py -march=native -DUSE_BOOST_SIMD -o $(name)_simd.so

clean:
	rm -f *.so

perf: perf_numpy perf_pythran perf_native perf_native_omp perf_omp perf_simd

perf_numpy:
	## numpy
	python -m perf timeit -s '$(setup) from $(name) import myfunc' 'myfunc(a)'

perf_pythran:
	## default
	python -m perf timeit -s '$(setup) from $(name)_default import myfunc' 'myfunc(a)'

perf_native:
	## native
	python -m perf timeit -s '$(setup) from $(name)_native import myfunc' 'myfunc(a)'

perf_omp:
	## openmp
	python -m perf timeit -s '$(setup) from $(name)_omp import myfunc' 'myfunc(a)'
	OMP_NUM_THREADS=1 python -m perf timeit -s '$(setup) from $(name)_omp import myfunc' 'myfunc(a)' --inherit-environ=OMP_NUM_THREADS

time_omp:
	time python -c '$(setup) from $(name)_omp import myfunc as f; [1 for i in range(1000) if f(a)[0, 0]]'
	OMP_NUM_THREADS=1 time python -c '$(setup) from $(name)_omp import myfunc as f; [1 for i in range(1000) if f(a)[0, 0]]'

time_native:
	time python -c '$(setup) from $(name)_native import myfunc as f; [1 for i in range(1000) if f(a)[0, 0]]'

perf_native_omp:
	## native_openmp
	python -m perf timeit -s '$(setup) from $(name)_native_omp import myfunc' 'myfunc(a)'

perf_simd:
	## simd
	python -m perf timeit -s '$(setup) from $(name)_simd import myfunc' 'myfunc(a)'
