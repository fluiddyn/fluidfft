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

Benchmarks on Occigen
---------------------

`Occigen is a GENCI-CINES HCP cluster <https://www.top500.org/system/178465>`_.

.. figure:: tmp/fig_occigen_384x1152x1152.png
   :scale: 150 %
   :alt: Benchmarks fig_occigen_384x1152x1152.png

   Speedup computed from the median of the elapsed times for 3d fft
   (384x1152x1152, left: fft and right: ifft) on Occigen.

The fastest methods are fftw1d (which is limited to 96 cores) and p3dfft.

The benchmark is not sufficiently accurate to measure the cost of calling the
functions from Python (difference between continuous and dashed lines,
i.e. between pure C++ and the "as_arg" Python method) and even the creation
of the numpy array (difference between the dashed and the dotted line,
i.e. between the "as_arg" and the "return" Python methods).

.. figure:: tmp/fig_occigen_1152x1152x1152.png
   :scale: 90 %
   :alt: Benchmarks fig_occigen_1152x1152x1152.png

   Speedup computed from the median of the elapsed times for 3d fft
   (1152x1152x1152, left: fft and right: ifft) on Occigen.

For this resolution, the fftw1d is also the fastest method when using only few
cores and it can not be used for more that 192 cores. The faster library when
using more cores is also p3dfft.

Benchmarks on Beskow
--------------------

`Beskow is a Cray machine running Intel processors
<https://www.pdc.kth.se/hpc-services/computing-systems>`_.

.. figure:: tmp/fig_beskow_384x1152x1152.png
   :scale: 90 %
   :alt: Benchmarks fig_beskow_384x1152x1152.png

   Speedup computed from the median of the elapsed times for 3d fft
   (384x1152x1152, left: fft and right: ifft) on Beskow.


Benchmarks on a `LEGI <http://www.legi.grenoble-inp.fr>`_ cluster
-----------------------------------------------------------------

We run some benchmarking in Cluster8 (2015, 12 nodes Xeon DELL C6320, 20 cores
per node).

.. figure:: tmp/fig_legi_cluster8_320x640x640.png
   :scale: 90 %
   :alt: Benchmarks 3d fft at LEGI on cluster8.

   Speedup computed from the median of the elapsed times for 3d fft
   (320x640x640, left: fft and right: ifft) at LEGI on cluster8.




..
   .. figure:: tmp/fig_legi_cluster7_2d.png
      :scale: 90 %
      :alt: Benchmarks 2d fft at LEGI on cluster7.

      Speedup computed from the median of the elapsed times for 2d fft
      (1024x1024, left: fft and right: ifft) at LEGI on cluster7 (2014, 16
      nodes Xeon DELL C6220, 16 cores per node). We see that the scaling is not
      far from linear for intra-node computation. In contrast, the speedup is
      really bad for computations involving inter-node computation.

      The benchmark is not sufficiently accurate to measure the cost of calling
      the functions from Python. The method fft2dmpiwithfftw1d is slower and
      seems less regular.

..
   .. figure:: tmp/fig_legi_cluster8_2d.png
      :scale: 90 %
      :alt: Benchmarks 2d fft at LEGI on cluster8.

      Same as previous figure but for "cluster8" (2015, 12 nodes Xeon DELL C6320,
      20 cores per node).
