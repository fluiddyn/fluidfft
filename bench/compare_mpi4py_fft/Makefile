TIMEIT:=python -m pyperf timeit --fast -s
# TIMEIT:=python -m pyperf timeit -q -s
NPROC:=2

FFT:='from bench import fft as transform;'
IFFT:='from bench import ifft as transform;'
INIT2D:='from bench import init2d as init, create_arrayX, create_arrayK;'
INIT3D:='from bench import init3d as init, create_arrayX, create_arrayK;'
SLAB:='o = init(slab=True); u = create_arrayX(o); u_hat = create_arrayK(o);'
PENCIL:='o = init(slab=False); u = create_arrayX(o); u_hat = create_arrayK(o);'
CODE:='transform(o, u, u_hat)'

all: bench2d bench3dslab bench3dpencil

install:
	pip install numpy perf
	FFTW_ROOT=/usr pip install mpi4py-fft

bench2d:
	@echo "---------------- 2D MPI benchmarks ----------------"
	@printf "fft: "
	@mpirun -np $(NPROC) $(TIMEIT) $(FFT)$(INIT2D)$(SLAB) $(CODE)
	@printf "ifft: "
	@mpirun -np $(NPROC) $(TIMEIT) $(IFFT)$(INIT2D)$(SLAB) $(CODE)

bench3dslab:
	@echo "------ 3D MPI slab decomposition benchmarks -------"
	@printf "fft: "
	@mpirun -np $(NPROC) $(TIMEIT) $(FFT)$(INIT3D)$(SLAB) $(CODE)
	@printf "ifft: "
	@mpirun -np $(NPROC) $(TIMEIT) $(IFFT)$(INIT3D)$(SLAB) $(CODE)


bench3dpencil:
	@echo "----- 3D MPI pencil decomposition benchmarks -----"
	@printf "fft: "
	@mpirun -np $(NPROC) $(TIMEIT) $(FFT)$(INIT3D)$(PENCIL) $(CODE)
	@printf "ifft: "
	@mpirun -np $(NPROC) $(TIMEIT) $(IFFT)$(INIT3D)$(PENCIL) $(CODE)
