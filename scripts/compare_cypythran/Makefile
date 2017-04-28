
all:
	# pythran -DUSE_BOOST_SIMD -march=native grad_pythran.py
	# pythran -fopenmp -march=corei7-avx grad_pythran.py
	# pythran -fopenmp grad_pythran.py
	pythran grad_pythran.py
	python setup.py build_ext --inplace

perf:
	# python
	python -m perf timeit -s 'from bench import gradfft_from_fft_py as g, f_fft, KX, KY' 'g(f_fft, KX, KY)'
	# pythran
	python -m perf timeit -s 'from bench import gradfft_from_fft as g, f_fft, KX, KY' 'g(f_fft, KX, KY)'
	# cython with @cython.boundscheck(False) @cython.wraparound(False)
	python -m perf timeit -s 'from bench import gradfft_from_fft_nocheck_cy as g, f_fft, KX, KY' 'g(f_fft, KX, KY)'
	# cython without @cython.boundscheck(False) @cython.wraparound(False)
	python -m perf timeit -s 'from bench import gradfft_from_fft_check_cy as g, f_fft, KX, KY' 'g(f_fft, KX, KY)'
