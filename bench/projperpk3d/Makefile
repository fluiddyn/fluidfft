
all: proj_default.so proj_native.so proj1_default.so proj2_default.so proj_omp.so

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

code = 'func(c0, c1, c2, a0, a1, a2, a3)'
end=as func, c0, c1, c2, a0, a1, a2, a3

perf: perfpython perfnative

perfnative:
	python -m pyperf timeit -s \
	  'from bench import proj_native $(end)' $(code)

perfdefault:
	python -m pyperf timeit -s \
	  'from bench import proj_default $(end)' $(code)

perfdefault1:
	python -m pyperf timeit -s \
	  'from bench import proj1_default $(end)' $(code)

perfdefault2:
	python -m pyperf timeit -s \
	  'from bench import proj2_default $(end)' $(code)

perffft:
	python -m pyperf timeit -s \
	  'from bench import proj_fft $(end)' $(code)

perfomp:
	python -m pyperf timeit -s \
	  'from bench import proj_omp $(end)' $(code)
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
