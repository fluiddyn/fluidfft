
name=mymod

all: $(name)_default.so $(name)_simd.so

$(name)_default.so: $(name).py
	pythran -v $(name).py -o $(name)_default.so

$(name)_native.so: $(name).py
	pythran -v $(name).py -march=native -o $(name)_native.so

$(name)_simd.so: $(name).py
	pythran -v $(name).py -march=native -DUSE_XSIMD -o $(name)_simd.so

$(name)_omp.so: $(name).py
	pythran -v $(name).py -fopenmp -o $(name)_omp.so

clean:
	rm -f *.so

perf2d: perfdefault2d perfsimd2d
perf3d: perfdefault3d perfsimd3d

code='func(arr)'


perfpython2d: all
	## numpy 2d (no loop)
	@python -m pyperf timeit -s \
	  'from bench import myfunc_py as func, f2d as arr' $(code)

perfdefault2d: all
	## default 2d (no loop)
	@python -m pyperf timeit -s \
	  'from bench import myfunc_default as func, f2d as arr' $(code)
	# default 2d (explicit loops)
	@python -m pyperf timeit -s \
	  'from bench import myfunc_loops2d_default as func, f2d as arr' $(code)

perfsimd2d: $(name)_simd.so
	# simd 2d (no loop)
	@python -m pyperf timeit -s \
	  'from bench import myfunc_simd as func, f2d as arr' $(code)
	# simd 2d (explicit loops)
	@python -m pyperf timeit -s \
	  'from bench import myfunc_loops2d_simd as func, f2d as arr' $(code)


perfpython3d: all
	## numpy 3d (no loop)
	@python -m pyperf timeit -s \
	  'from bench import myfunc_py as func, f3d as arr' $(code)

perfdefault3d: all
	# default 3d (no loop)
	@python -m pyperf timeit -s \
	  'from bench import myfunc_default as func, f3d as arr' $(code)
	# default 3d (explicit loops)
	@python -m pyperf timeit -s \
	  'from bench import myfunc_loops3d_default as func, f3d as arr' $(code)

perfsimd3d: all
	# simd 3d (no loop)
	@python -m pyperf timeit -s \
	  'from bench import myfunc_simd as func, f3d as arr' $(code)
	# simd 3d (explicit loops)
	@python -m pyperf timeit -s \
	  'from bench import myfunc_loops3d_simd as func, f3d as arr' $(code)
