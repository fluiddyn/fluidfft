
all: perf perffortran

proj_default.so: proj.py
	pythran -v proj.py -DUSE_XSIMD -o proj_default.so

proj_native.so: proj.py
	pythran -v proj.py -march=native -DUSE_XSIMD -o proj_native.so

proj_omp.so: proj.py
	pythran -v proj.py -march=native -DUSE_XSIMD -fopenmp -o proj_omp.so

clean: cleanfortran
	rm -f *.so

cleanfortran:
	rm -f bench_proj_fortran.out

code = 'func(c0, c1, c2, a0, a1, a2, a3)'
end=as func, c0, c1, c2, a0, a1, a2, a3

perf: perfpython perfnative

perfnative: proj_native.so
	python -m pyperf timeit -s \
	  'from bench import proj_native $(end)' $(code)
	python -m pyperf timeit -s \
	  'from bench import proj_loop_native $(end)' $(code)

perfdefault: proj_default.so
	python -m pyperf timeit -s \
	  'from bench import proj_default $(end)' $(code)
	python -m pyperf timeit -s \
	  'from bench import proj_loop_default $(end)' $(code)

# perffft:
# 	python -m pyperf timeit -s \
# 	  'from bench import proj_fft $(end)' $(code)

perfomp:
	OMP_NUM_THREADS=2 python -m pyperf timeit -s \
	  'from bench import proj_omp $(end)' $(code) \
	  --inherit-environ=OMP_NUM_THREADS
	OMP_NUM_THREADS=1 python -m pyperf timeit -s \
	  'from bench import proj_omp $(end)' $(code) \
	  --inherit-environ=OMP_NUM_THREADS

perfpython:
	python -m pyperf timeit -s \
	  'from bench import proj_py $(end)' $(code)

perffortran: bench_proj_fortran.out
	./bench_proj_fortran.out

bench_proj_fortran.out: bench_proj_fortran.f90
	gfortran bench_proj_fortran.f90 -O3 -o bench_proj_fortran.out
