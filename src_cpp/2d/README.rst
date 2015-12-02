2d FFT
======

``make`` to build the program and then::

  mpirun -np 4 ./test_bench.out --N0=1024 --N1=1024


class FFT2DMPIWithFFTW1D (distributed memory - MPI)
---------------------------------------------------

The ' indicates the dimensions that are distributed over processes.

nx % 2 == ny % 2 == 0.

nkx = nx/2. This means that we forget the modes with ikx = nx/2, which
correspond to kx = deltakx * nx/2.

Based on 1d fft functions (real to complex and complex to complex).


s[iy', ix]       # shape == [ny, nx]

|
| fft 1d (real to complex)
|

sF1d[iy', ikx]   # shape == [ny, nkx]

|
| transpose
|

sF1d[ikx', iy]   # shape == [nkx, ny]

|
| fft 1d (complex to complex)
|

sF2d[ikx', iky]  # shape == [nkx, ny]


class FFT2DWithFFTW1D (with shared memory - Open MP)
----------------------------------------------------
