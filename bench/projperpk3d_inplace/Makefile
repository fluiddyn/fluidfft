
all: proj_default.so proj_native.so proj_omp.so

proj_default.so: proj.py
	pythran -v proj.py -o proj_default.so

proj_native.so: proj.py
	pythran -v proj.py -march=native -o proj_native.so

proj_omp.so: proj.py
	pythran -v proj.py -march=native -fopenmp -o proj_omp.so

clean: cleanfortran
	rm -f *.so

cleanfortran:
	rm -f bench_proj_fortran.out

code = 'func(c0, c1, c2, a0, a1, a2, a3)'
end=as func, c0, c1, c2, a0, a1, a2, a3

perf: perfpython perfnative

perfnative:
	python -m perf timeit -s \
	  'from bench import proj_native $(end)' $(code)

perfdefault:
	python -m perf timeit -s \
	  'from bench import proj_default $(end)' $(code)

perffft:
	python -m perf timeit -s \
	  'from bench import proj_fft $(end)' $(code)

perfomp:
	OMP_NUM_THREADS=2 python -m perf timeit -s \
	  'from bench import proj_omp $(end)' $(code) \
	  --inherit-environ=OMP_NUM_THREADS
	OMP_NUM_THREADS=1 python -m perf timeit -s \
	  'from bench import proj_omp $(end)' $(code) \
	  --inherit-environ=OMP_NUM_THREADS

perfpython:
	python -m perf timeit -s \
	  'from bench import proj_py $(end)' $(code)

perffortran: bench_proj_fortran.out
	./bench_proj_fortran.out

bench_proj_fortran.out: bench_proj_fortran.f90
	gfortran bench_proj_fortran.f90 -O3 -o bench_proj_fortran.out
