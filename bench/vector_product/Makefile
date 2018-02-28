
name=vectprod

all: $(name)_default.so $(name)_native.so $(name)_omp.so

$(name)_default.so: $(name).py
	pythran -v $(name).py -o $(name)_default.so

$(name)_native.so: $(name).py
	pythran -v $(name).py -march=native -o $(name)_native.so

$(name)_omp.so: $(name).py
	pythran -v $(name).py -march=native -fopenmp -o $(name)_omp.so

clean: cleanfortran
	rm -f *.so

cleanfortran:
	rm -f bench_$(name)_fortran.out

code = 'func(a0, a1, a2, a3, a4, a5)'
arrays=a0, a1, a2, a3, a4, a5

perf: perfpython perfnative

perfnative:
	python -m perf timeit -s \
	  'from bench import $(name)_native as func, $(arrays)' $(code)

perfdefault:
	python -m perf timeit -s \
	  'from bench import $(name)_default as func, $(arrays)' $(code)

perffft:
	python -m perf timeit -s \
	  'from bench import $(name)_fft as func, $(arrays)' $(code)

perfomp:
	python -m perf timeit -s \
	  'from bench import $(name)_omp as func, $(arrays)' $(code)
	OMP_NUM_THREADS=1 python -m perf timeit -s \
	  'from bench import $(name)_omp as func, $(arrays)' $(code) \
	  --inherit-environ=OMP_NUM_THREADS

perfpython:
	python -m perf timeit -s \
	  'from bench import $(name)_py as func, $(arrays)' $(code)


perfnative1:
	python -m perf timeit -s \
	  'from bench import $(name)1_native as func, $(arrays)' $(code)

perfdefault1:
	python -m perf timeit -s \
	  'from bench import $(name)1_default as func, $(arrays)' $(code)

perfomp1:
	python -m perf timeit -s \
	  'from bench import $(name)1_omp as func, $(arrays)' $(code)
	OMP_NUM_THREADS=1 python -m perf timeit -s \
	  'from bench import $(name)1_omp as func, $(arrays)' $(code) \
	  --inherit-environ=OMP_NUM_THREADS

perfpython1:
	python -m perf timeit -s \
	  'from bench import $(name)1_py as func, $(arrays)' $(code)


perfnative2:
	python -m perf timeit -s \
	  'from bench import $(name)2_native as func, $(arrays)' $(code)

perfdefault2:
	python -m perf timeit -s \
	  'from bench import $(name)2_default as func, $(arrays)' $(code)

perfomp2:
	python -m perf timeit -s \
	  'from bench import $(name)2_omp as func, $(arrays)' $(code)
	OMP_NUM_THREADS=1 python -m perf timeit -s \
	  'from bench import $(name)2_omp as func, $(arrays)' $(code) \
	  --inherit-environ=OMP_NUM_THREADS

perfpython2:
	python -m perf timeit -s \
	  'from bench import $(name)2_py as func, $(arrays)' $(code)


perffortran: bench_$(name)_fortran.out
	./bench_$(name)_fortran.out

bench_$(name)_fortran.out: bench_$(name)_fortran.f90
	gfortran bench_$(name)_fortran.f90 -O3 -o bench_$(name)_fortran.out