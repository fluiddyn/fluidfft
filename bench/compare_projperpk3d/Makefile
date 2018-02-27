
proj_setup=proj_setup.cpython-36m-x86_64-linux-gnu.so

all: proj_default.so proj_native.so proj_omp.so $(proj_setup) proj1_default.so proj2_default.so

proj_default.so: proj.py
	pythran -v proj.py -o proj_default.so

proj1_default.so: proj1.py
	pythran -v proj1.py -o proj1_default.so

proj2_default.so: proj2.py
	pythran -v proj2.py -o proj2_default.so

proj_native.so: proj.py
	pythran -v proj.py -march=native -o proj_native.so

proj_omp.so: proj.py
	pythran -v proj.py -march=native -fopenmp -o proj_omp.so

$(proj_setup): proj.py setup.py
	python setup.py build_ext --inplace

clean:
	rm -f *.so

code = 'func(arr_c, arr_c, arr_c, arr, arr, arr, arr)'

perf: perfpython perfpythran perffft

perfnative:
	python -m perf timeit -s \
	  'from bench import proj_native as func, arr_c, arr' $(code)

perfdefault:
	python -m perf timeit -s \
	  'from bench import proj_default as func, arr_c, arr' $(code)

perfdefault1:
	python -m perf timeit -s \
	  'from bench import proj1_default as func, arr_c, arr' $(code)

perfdefault2:
	python -m perf timeit -s \
	  'from bench import proj2_default as func, arr_c, arr' $(code)

perfsetup:
	python -m perf timeit -s \
	  'from bench import proj_setup as func, arr_c, arr' $(code)

perfomp:
	python -m perf timeit -s \
	  'from bench import proj_omp as func, arr_c, arr' $(code)
	OMP_NUM_THREADS=1 python -m perf timeit -s \
	  'from bench import proj_omp as func, arr_c, arr' $(code) \
	  --inherit-environ=OMP_NUM_THREADS

perffft:
	python -m perf timeit -s \
	  'from bench import proj_fft as func, arr_c, arr' $(code)

perfpython:
	python -m perf timeit -s \
	  'from bench import proj_py as func, arr_c, arr' $(code)
