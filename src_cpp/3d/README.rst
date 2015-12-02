3d FFT with distributed memory (MPI)
====================================

Different libraries

- fftw3 and fftw3-mpi
  
- pfft https://github.com/mpip/pfft

  (with a Python binding https://github.com/rainwoodman/pfft-python)

- p3dfft https://github.com/sdsc/p3dfft

- 2decomp&FFT http://www.2decomp.org

  (seems to be dead?)


Planes
------

Based on 1d and 2d fft functions (real to complex and complex to complex).

The ' indicates the dimensions that are distributed over processes.

We take nx % 2 == ny % 2 == nz % 2 == 0.

nkx = nx/2
nky = ny
nyz = nz


s[iz', iy, ix]      # shape == [nz_loc, ny, nx]

|
| fft 2d (real to complex)
|

sF2d[iz', iky, ikx]    # shape == [nz_loc, nky, nkx]

|
| transpose with mpi communication
|

sF2d[iky_loc', ikx, iz]    # shape == [nky_loc, nkx, nz]

|
| fft 1d (complex to complex)
|

sF3d[iky_loc', ikx, ikz]    # shape == [nky_loc, nkx, nkz]


Pencils
-------

Based on 1d fft functions (real to complex and complex to complex).

The ' indicates the dimensions that are distributed over processes.

We take nx % 2 == ny % 2 == nz % 2 == 0.

nkx = nx/2
nky = ny
nyz = nz


s[iz', iy', ix]

|
| fft 1d (real to complex)
|

sF1d[iz', iy', ikx]

|
| transpose with mpi communication
|

sF1d[iz', ikx', iy]

|
| fft 1d (complex to complex)
|

sF2d[iz', ikx', iky]

|
| transpose with mpi communication
|

sF2d[ikx', iky', iz]

|
| fft 1d (complex to complex)
|

sF3d[ikx', iky', ikz]
