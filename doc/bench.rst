.. _bench:

Benchmarking
============

Command-line utilities
----------------------

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

Benchmarks on `LEGI <http://www.legi.grenoble-inp.fr>`_ clusters
----------------------------------------------------------------

.. figure:: tmp/fig_legi_cluster7_2d.png
   :scale: 80 %
   :alt: Benchmarks at LEGI on cluster7.

   Speedup computed from the median of the elapsed times for 2d fft (1024x1024,
   left: fft and right: ifft) at LEGI on cluster7 (2014, 16 nodes Xeon DELL
   C6220, 16 cores per node). We see that the scaling is not far from linear
   for intra-node computation. In contrast, the speedup is really bad for
   computations involving inter-node computation. 

   The benchmark is not sufficiently accurate to measure the cost of calling
   the functions from Python. The method fft2dmpiwithfftw1d is slower and seems
   less regular.

.. figure:: tmp/fig_legi_cluster8_2d.png
   :scale: 80 %
   :alt: Benchmarks at LEGI on cluster8.

   Same as previous figure but for "cluster8" (2015, 12 nodes Xeon DELL C6320,
   20 cores per node).
