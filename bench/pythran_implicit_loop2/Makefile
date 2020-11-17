
name=mymod
nameso=$(name)_thran.so

# TIMEIT=python -m pyperf timeit --fast -q -s
TIMEIT=python -m pyperf timeit -q -s

# PYTHRAN=pythran -march=native -DUSE_BOOST_SIMD
PYTHRAN=pythran

all: $(nameso)

$(nameso): $(name).py
	$(PYTHRAN) -v $(name).py -o $(nameso)

clean:
	rm -f *.so


code='func(arr0, arr1)'


perf2d_float_py:
	### 2d, float, no loop, return
	@$(TIMEIT) \
	   'from bench import myfunc_ret_py as func, f2d as arr0; arr1 = arr0'\
           $(code)
	### 2d, float, no loop, inplace
	@$(TIMEIT) \
	   'from bench import myfunc_py as func, f2d as arr0; arr1 = arr0' $(code)

perf2d_float: $(nameso)
	### 2d, float, no loop, return
	@$(TIMEIT) \
	   'from bench import myfunc_ret as func, f2d as arr0; arr1 = arr0'\
           $(code)
	### 2d, float, no loop, inplace
	@$(TIMEIT) \
	   'from bench import myfunc as func, f2d as arr0; arr1 = arr0' $(code)
	### 2d, float, explicit loops, inplace
	@$(TIMEIT) \
	   'from bench import myfunc_loops2d as func, f2d as arr0; arr1 = arr0'\
           $(code)

perf2d_complex_py:
	### 2d, complex, no loop, return, py
	@$(TIMEIT) \
	   'from bench import myfunc_ret_py as func, f2d_c as arr0; arr1 = arr0'\
           $(code)
	### 2d, complex, no loop, inplace
	@$(TIMEIT) \
	   'from bench import myfunc_py as func, f2d_c as arr0; arr1 = arr0'\
	   $(code)

perf2d_complex: $(nameso)
	### 2d, complex, no loop, return
	@$(TIMEIT) \
	   'from bench import myfunc_ret as func, f2d_c as arr0; arr1 = arr0'\
           $(code)
	### 2d, complex,  no loop, inplace
	@$(TIMEIT) \
	   'from bench import myfunc as func, f2d_c as arr0; arr1 = arr0' $(code)
	### 2d, complex, explicit loops, inplace
	@$(TIMEIT) \
	   'from bench import myfunc_loops2d as func, f2d_c as arr0; arr1 = arr0'\
           $(code)


perf3d: perf3d_float_py perf3d_float perf3d_complex_py perf3d_complex


perf3d_float_py:
	### 3d, float, no loop, return, py
	@$(TIMEIT) \
	   'from bench import myfunc_ret_py as func, f3d as arr0; arr1 = arr0'\
           $(code)
	### 3d, float, no loop, inplace, py
	@$(TIMEIT) \
	   'from bench import myfunc_py as func, f3d as arr0; arr1 = arr0' $(code)

perf3d_float: $(nameso)
	### 3d, float, no loop, return
	@$(TIMEIT) \
	   'from bench import myfunc_ret as func, f3d as arr0; arr1 = arr0'\
           $(code)
	### 3d, float, no loop, inplace
	@$(TIMEIT) \
	   'from bench import myfunc as func, f3d as arr0; arr1 = arr0' $(code)
	### 3d, float, explicit loops reshape, inplace
	@$(TIMEIT) \
	   'from bench import myfunc_loops_reshape as func, f3d as arr0; arr1 = arr0'\
           $(code)
	### 3d, float, explicit loops, inplace
	@$(TIMEIT) \
	   'from bench import myfunc_loops3d as func, f3d as arr0; arr1 = arr0'\
           $(code)

perf3d_complex_py:
	### 3d, complex, no loop, return, py
	@$(TIMEIT) \
	   'from bench import myfunc_ret_py as func, f3d_c as arr0; arr1 = arr0'\
           $(code)
	### 3d, complex, no loop, inplace, py
	@$(TIMEIT) \
	   'from bench import myfunc_py as func, f3d_c as arr0; arr1 = arr0'\
	   $(code)

perf3d_complex: $(nameso)
	### 3d, complex, no loop, return
	@$(TIMEIT) \
	   'from bench import myfunc_ret as func, f3d_c as arr0; arr1 = arr0'\
           $(code)
	### 3d, complex, no loop, inplace
	@$(TIMEIT) \
	   'from bench import myfunc as func, f3d_c as arr0; arr1 = arr0'\
           $(code)
	### 3d, complex, explicit loops reshape, inplace
	@$(TIMEIT) \
	   'from bench import myfunc_loops_reshape as func, f3d_c as arr0; arr1 = arr0'\
           $(code)

	### 3d, complex, explicit loops, inplace
	@$(TIMEIT) \
	   'from bench import myfunc_loops3d as func, f3d_c as arr0; arr1 = arr0'\
           $(code)
