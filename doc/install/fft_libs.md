# Supported FFT libraries and their installation

## fftw3 (better to build with OpenMP enabled)

[FFTW](http://www.fftw.org/) (*Fastest Fourier Transform in the West*) is the
standard open-source library for discrete Fourier transform.

```{eval-rst}
.. todo::

   Try hybrid OpenMP + MPI...
```

## fftw3_mpi

[FFTW](http://www.fftw.org/) provide their parallel MPI implementation.

If a system-wide installation of FFTW with OpenMP and MPI is not available,
the user can make a local installation in a user-accessible directory. You can
utilize this bash script with some modifications to the first few lines to
install the FFTW library.

```{literalinclude} install_fftw.sh
:language: shell
```

## MKL library (FFT by intel)

There are wrappers to use the MKL library (FFT by intel) using the FFTW
API. See the file `site.cfg.mkl`.

## [pfft](https://github.com/mpip/pfft)

*"PFFT is a software library for computing massively parallel, fast Fourier
transformations on distributed memory architectures. PFFT can be understood as
a generalization of FFTW-MPI to multidimensional data decomposition. The
library is written in C and MPI."*

You may adapt the shell script given below to install pfft. Some other scripts
related to IBM clusters can be found [here](https://www-user.tu-chemnitz.de/~potts/workgroup/pippig/software.php.en#scripts).

```{literalinclude} install_pfft.sh
:language: shell
```

This script can be downloaded with:

```
wget https://foss.heptapod.net/fluiddyn/fluidfft/-/raw/branch/default/doc/install/install_pfft.sh
```

## [p3dfft](https://www.p3dfft.net/)

*"Parallel Three-Dimensional Fast Fourier Transforms, dubbed P3DFFT, is a
library for large-scale computer simulations on parallel platforms. \[...\]
P3DFFT uses 2D, or pencil, decomposition. This overcomes an important
limitation to scalability inherent in FFT libraries implementing 1D (or slab)
decomposition: the number of processors/tasks used to run this problem in
parallel can be as large as N2, where N is the linear problem size."*

To use p3dfft with python binding, we need a shared library (.so) and therefore,
we use a modified version of the package. Please follow the process detailed
in the INSTALL file of this fork of the official p3dfft github repository :
[CyrilleBonamy/p3dfft](https://github.com/CyrilleBonamy/p3dfft). The process
can be summarized as follows :

It can be convenient to link p3dfft with fftw3. A single fftw directory with
lib and include directories must exist for this purpose.

For example, on Debian system, you can do that with:

```
ROOTFFTW=$HOME/opt/fft_gcc
mkdir -p $ROOTFFTW/include
mkdir -p $ROOTFFTW/lib
cp /usr/include/fftw* $ROOTFFTW/include
cp /usr/lib/x86_64-linux-gnu/libfftw3* $ROOTFFTW/lib
```

And next you can compile p3dfft with the following command:

```
CC=mpicc CCLD=mpif90 ./configure --enable-fftw --with-fftw=$ROOTFFTW \
    --prefix=/opt/p3dfft/2.7.5
```

You may adapt the shell script given below to automate this process :

```{literalinclude} install_p3dfft.sh
:language: shell
```

This script can be downloaded with:

```sh
wget https://foss.heptapod.net/fluiddyn/fluidfft/-/raw/branch/default/doc/install/install_p3dfft.sh
```

## cuda

Modify fftw.h ! <https://github.com/FFTW/fftw3/issues/18>

```{eval-rst}
.. todo::

   How can I install Cuda? Link? Advice?
```

# Unsupported libraries

## [openfft](http://www.openmx-square.org/openfft/)

There is no class of fluidfft to use the library openfft because it does not
provide a function for inverse fft.

## [2decomp](http://www.2decomp.org)

There is no class of fluidfft to use the library 2decomp because we didn't
manage to build their shared libraries.
