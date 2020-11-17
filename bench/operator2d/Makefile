
name=util

setup=import numpy as np; a = np.ones([1000, 1000]);

extbin=$(shell python -c "from distutils.sysconfig import get_config_var; print(get_config_var('EXT_SUFFIX'))")

all: default/$(name)_default$(extbin) native/$(name)_native$(extbin) openmp/$(name)_openmp$(extbin)

clean:
	rm -f */*$(extbin)

cleanall: clean
	rm -rf default native openmp util.py

$(name).py: ../../fluidfft/fft2d/util_pythran.py
	cp ../../fluidfft/fft2d/util_pythran.py $(name).py

default/$(name)_default$(extbin): $(name).py
	mkdir -p default
	echo 'from . import util' > default/__init__.py
	pythran $(name).py && mv $(name).so default/$(name)$(extbin)

native/$(name)_native$(extbin): $(name).py
	mkdir -p native
	echo 'from . import util' > native/__init__.py
	pythran $(name).py -march=native && mv $(name).so native/$(name)$(extbin)

openmp/$(name)_openmp$(extbin): $(name).py
	mkdir -p openmp
	echo 'from . import util' > openmp/__init__.py
	pythran $(name).py -fopenmp && mv $(name).so openmp/$(name)$(extbin)

# $(name)_native_openmp$(extbin): $(name).py
# 	pythran $(name).py -march=native -fopenmp && mv $(name).so $(name)_native_openmp$(extbin)

# $(name)_simd$(extbin): $(name).py
# 	pythran $(name).py -march=native -DUSE_BOOST_SIMD && mv $(name).so $(name)_simd$(extbin)

# perf:
# 	## numpy
# 	python -m pyperf timeit -s '$(setup) from $(name) import myfunc' 'myfunc(a)'
# 	## default
# 	python -m pyperf timeit -s '$(setup) from $(name)_default import myfunc' 'myfunc(a)'
# 	## native
# 	python -m pyperf timeit -s '$(setup) from $(name)_native import myfunc' 'myfunc(a)'
# 	## native_openmp
# 	python -m pyperf timeit -s '$(setup) from $(name)_native_openmp import myfunc' 'myfunc(a)'
# 	## openmp
# 	python -m pyperf timeit -s '$(setup) from $(name)_openmp import myfunc' 'myfunc(a)'
# 	## simd
# 	python -m pyperf timeit -s '$(setup) from $(name)_simd import myfunc' 'myfunc(a)'
