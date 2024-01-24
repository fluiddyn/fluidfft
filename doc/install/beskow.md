# Installation on the HPC cluster Beskow

[Beskow is a Cray machine running Intel processors](https://www.pdc.kth.se/hpc-services/computing-systems). The procedure
described below could arguably be one of the most complex methods due to the
obsolete collection of modules and the use of wrappers which is designed to
enable cross-compilation in all three compilation environments (Prg-cray,
Prg-intel and Prg-gnu). Here we have managed to install FluidFFT in Prg-intel
environment.

Load necessary modules:

```
module load gcc/6.1.0
module swap PrgEnv-cray PrgEnv-intel
module swap intel intel/18.0.0.128
module load mercurial
module load cray-hdf5-parallel
```

Ensure that `gcc` is loaded before `intel`. This is important so that Intel
libraries get prepended to `LD_LIBRARY_PATH` and should supersede equivalent
GCC libraries, since we intent to use Intel compilers.

Installing additional packages using Anaconda provided in Beskow can be
troublesome. Install Miniconda instead, setup a virtual environment and install
necessary python packages as described here:

```bash
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

module load craype-hugepages2M  # cray-hdf5-parallel

conda install ipython
conda install numpy
conda install h5py
conda install scipy matplotlib
conda install -c conda-forge pyfftw

pip install cython
export CRAYPE_LINK_TYPE=dynamic
module load craype-hugepages2M
CC=cc MPICC=cc pip install mpi4py -v --no-deps --no-binary mpi4py
module rm craype-hugepages2M

pip install pythran colorlog nose
```

Run a few tests:

```
salloc -A <project> -N 1 -t 10:00
export CRAY_ROOTFS=DSL
export OMP_NUM_THREADS=1
aprun -n 1 python -c 'import numpy; numpy.test()'
aprun -n 1 python -c 'import h5py; h5py.run_tests()'
aprun -n 2 python -m unittest fluiddyn.util.test.test_mpi
```

Create the file `~/.pythranrc` with:

```
[pythran]
complex_hook = True

[compiler]
CC=gcc
CXX=g++
```

However do not run `module load fftw` as the latest version available now
(2017-11-19) `3.3.4` and requires patching to use pfft. Also we need recent
versions of GNU Autotools to properly build the packages, again not supplied
with Beskow! We start with:

```{literalinclude} beskow_install_autotools.sh
:language: shell
```

This will install GNU stow, autoconf, automake, libtool. GNU stow will manage
all local installation henceforth as demonstrated below. This is useful to keep
the local directory clean, if something goes wrong, and to manage versions.

```bash
stow -v <directory>  # symlinks all files and directories to one level above
stow -Dv <directory>  # deletes all symlinks made before
```

:::{warning}
Manual stow maybe required when run on the compute node due to Perl not
being able to access the home directory.
:::

Now let us proceed to build fftw3, p3dfft and pfft. For some reason, the cross-
compilation wrappers do not work or fail to link when used with FFT libraries.
Therefore we explicitly use Intel compilers in the following steps:

:::{warning}
It is imperative to compile all the tools and FFT libraries in the compute
node. This is because as I write now (Feb 2018) Beskow uses different CPU
architectures in the login nodes / frontend (Sandy Bridge) compared to
compute nodes (Haskell). We know for sure this can affect FFTW compilation,
just to be sure, do everything on the compute node.
:::

For fftw3:

```{literalinclude} beskow_install_fftw.sh
:language: shell
```

For p3dfft:

```{literalinclude} beskow_install_p3dfft.sh
:language: shell
```

For pfft:

```{literalinclude} beskow_install_pfft.sh
:language: shell
```

Ensure that the stowed library location has been prepended to
`LD_LIBRARY_PATH`. Finally, we can install fluidfft:

```
hg clone https://foss.heptapod.net/fluiddyn/fluidfft
cd fluidfft
```

Activate the virutal environment. Copy `site.cfg.default` to `site.cfg`
within fluidfft directory or `~/.fluidfft-site.cfg` with all libraries
mentioned above set like:

```
use = True
dir = /cfs/klemming/nobackup/u/user
include_dir =
library_dir =
```

Install FluidFFT with:

```
export CRAYPE_LINK_TYPE=dynamic
module load craype-hugepages2M
MPICXX=CC LDSHARED="CC -shared" python setup.py develop
module rm craype-hugepages2M
```

Test your installation by allocating a compute node as follows:

```
salloc -A <project> -N 1 -t 05:00
export CRAY_ROOTFS=DSL
export OMP_NUM_THREADS=1
aprun -n 1 python -m unittest discover
aprun -n 2 python -m unittest discover
```
