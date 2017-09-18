
name=mymod

setup=import numpy as np; a = np.ones([1000, 1000]);

clean:
	rm -f *.so

compile:
	pythran $(name).py -o $(name)_default.so
	pythran $(name).py -march=native -o $(name)_native.so
	pythran $(name).py -march=native -fopenmp -o $(name)_native_openmp.so
	pythran $(name).py -fopenmp -o $(name)_openmp.so
	pythran $(name).py -march=native -DUSE_BOOST_SIMD -o $(name)_simd.so

perf:
	## numpy
	python -m perf timeit -s '$(setup) from $(name) import myfunc' 'myfunc(a)'
	## default
	python -m perf timeit -s '$(setup) from $(name)_default import myfunc' 'myfunc(a)'
	## native
	python -m perf timeit -s '$(setup) from $(name)_native import myfunc' 'myfunc(a)'
	## native_openmp
	python -m perf timeit -s '$(setup) from $(name)_native_openmp import myfunc' 'myfunc(a)'
	## openmp
	python -m perf timeit -s '$(setup) from $(name)_openmp import myfunc' 'myfunc(a)'
	## simd
	python -m perf timeit -s '$(setup) from $(name)_simd import myfunc' 'myfunc(a)'
