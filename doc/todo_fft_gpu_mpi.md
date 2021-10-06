# Description project library C++ FFT MPI+GPU

## Context

### fluidsim and fluidfft

We develop and use for HPC simulations the program
[fluidsim](https://foss.heptapod.net/fluiddyn/fluidsim) and the library
[fluidfft](https://foss.heptapod.net/fluiddyn/fluidfft).

Fluidsim is a Python package to run simulations of fluids. It is a framework to
define solvers but it is specialized on pseudo-spectral simulations based on
Fast Fourier Transforms (FFT).

Fluidsim uses fluidfft to perform the FFTs with different libraries. fluidfft
is written in ugly C++, in Cython and in Python. Parallelization is done in
Fluidsim and Fluidfft with MPI. One starts (with `mpirun`) as much processes as
the number of cores we will use.

Fluidfft is mostly a wrapper around other FFT libraries. However, there are
also simple implementations of MPI FFT classes for which the memory is
distributed over different processes. In particular I think the class
implemented in
https://foss.heptapod.net/fluiddyn/fluidfft/-/blob/branch/default/src_cpp/3d/fft3dmpi_with_fftw1d.cpp
is very interesting for this work. We see how a 3D FFT is only a serie of 3 1D
FFTs. We can see how the data is distributed in blocks and which MPI
communications are needed to do the computations of the 1D FFTs over the 3
dimensions.

### GHOST

Another code to run pseudo-spectral simulations based on FFT is
[GHOST](https://github.com/pmininni/GHOST). It is written in Fortran and has
been optimized to run efficient simulations on very large clusters.

GHOST is better than Fluidsim for coupling MPI and OpenMP parallelization
methods, so that the parallelization strategy is more tied to the architecture
of clusters. For example, one can use 1 process per CPU and use OpenMP for
intra-CPU parallelization (to use the different cores of the CPU). However,
there is nothing which really prevents Fluidsim to do the same.

### FFT parallelized with MPI and accelerated with GPU boards

There is one thing that GHOST can do and that Fluidsim cannot do at all for
now. GHOST is able to use GPU boards in nodes to accelerate FFT parallelized
with MPI. The code in GHOST is there:
https://github.com/pmininni/GHOST/blob/master/3D/src/fftp-cu/fftp3D.fpp

The simplest case to think about this is on 1 computer with 1 GPU. We want to
be able to compute FFT with distributed memory and that the different MPI
processes delegate the computation of 1d FFT to the GPU. Of course, if there
are 2 GPU boards, the processes should be able to use the 2 boards.

On clusters, we want to be able to use several nodes with their GPU. So the
processes have to delegate the computation of the 1D FFTs to the local GPUs
available on their node.

I tend to think that an implementation of such GPU accelerated MPI parallelized
FFTs could be very similar to what we already have in
3d/fft3dmpi_with_fftw1d.cpp except that the FFT plans have to use one GPU
board. Of course we will need a lot of transfers between the RAM and the GPU. I
think it is reasonable to limit ourself to cases for which one array for one
process can fit in the GPU memory.

## What we would like to have

### Final goal: a fluidfft class doing parallel FFT accelerated with GPU

I don't except to have this at the end of this work with QuantStack! But let's
see.

### Possible intermediate step

xtensor-fft-mpi-gpu ? That I could then use in fluidfft?

### CUDA or OpenCL

Of course CUDA would be the easy track. Note that we already have a CUDA
accelerated sequential FFT class in Fluidfft (old unused code).

However, I heard very good thing about OpenCL so it could be interesting to
investigate if we could use this technology.
