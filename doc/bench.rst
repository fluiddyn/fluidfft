.. _bench:

Benchmarking
============

One of the advantage of using fluidfft is to be able to use the fastest fft
library for a particular problem and in a particular (super)computer.

We provide command-line utilities to easily run and analyze benchmarks. You can
for example run the commands::

  fluidfft-bench -h

  # 2d
  fluidfft-bench 1024 768
  fluidfft-bench 1024 -d 2
  mpirun -np 2 fluidfft-bench 1024 -d 2

  # 3d
  fluidfft-bench 32 48 64
  fluidfft-bench 128 -d 3
  mpirun -np 2 fluidfft-bench 128 -d 3

Once you have run many benchmarks (to get statistics) for different numbers of
processes (if you want to use MPI), you can analyze the results for example
with::

  fluidfft-bench-analysis 1024 -d 2
