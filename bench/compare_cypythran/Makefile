
all:
	pythran -v -DUSE_BOOST_SIMD -march=native grad_pythran.py -o grad_simd.so
	pythran -v -fopenmp grad_pythran.py -o grad_omp.so
	pythran -v -march=native grad_pythran.py
	python setup.py build_ext --inplace

perf: perfpython perfcython perfpythran perfsimd perfomp

perfpython:
	# python
	python -m pyperf timeit -s \
	  'from bench import grad_py as g, f_fft, KX, KY' 'g(f_fft, KX, KY)'

perfcython:
	# cython with @cython.boundscheck(False) @cython.wraparound(False)
	python -m pyperf timeit -s \
	  'from bench import grad_cy_nocheck as g, f_fft, KX, KY' 'g(f_fft, KX, KY)'

perfcythoncheck:
	# cython without @cython.boundscheck(False) @cython.wraparound(False)
	python -m pyperf timeit -s \
	'from bench import grad_cy_check as g, f_fft, KX, KY' 'g(f_fft, KX, KY)'

perfpythran:
	# pythran
	python -m pyperf timeit -s \
	  'from bench import grad_pythran as g, f_fft, KX, KY' 'g(f_fft, KX, KY)'

perfsimd:
	# SIMD
	python -m pyperf timeit -s \
	  'from bench import grad_simd as g, f_fft, KX, KY' 'g(f_fft, KX, KY)'

perfomp:
	# OpenMP
	python -m pyperf timeit -s \
	  'from bench import grad_omp as g, f_fft, KX, KY' 'g(f_fft, KX, KY)'
