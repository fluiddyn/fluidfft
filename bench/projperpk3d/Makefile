
all: proj_default.so proj_native.so proj_omp.so proj1_default.so proj2_default.so

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

clean: cleanfortran
	rm -f *.so

cleanfortran:
	rm -f bench_proj_fortran.out

code = 'func(arr_c, arr_c, arr_c, arr, arr, arr, arr)'

perf: perfpython perfnative

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

perfomp:
	python -m perf timeit -s \
	  'from bench import proj_omp as func, arr_c, arr' $(code)
	OMP_NUM_THREADS=1 python -m perf timeit -s \
	  'from bench import proj_omp as func, arr_c, arr' $(code) \
	  --inherit-environ=OMP_NUM_THREADS

perfpython:
	python -m perf timeit -s \
	  'from bench import proj_py as func, arr_c, arr' $(code)

perffortran: bench_proj_fortran.out
	./bench_proj_fortran.out

bench_proj_fortran.out: bench_proj_fortran.f90
	gfortran bench_proj_fortran.f90 -O3 -o bench_proj_fortran.out
