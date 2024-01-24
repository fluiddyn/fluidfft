# Installation on the HPC cluster Froggy (Ciment, UGA)

[Froggy is a local HPC cluster in Grenoble](https://ciment.ujf-grenoble.fr/wiki-pub/index.php/Hardware:Froggy).

:::{warning}
Until now (2018-02), we have not been able to install fluidfft on Froggy!
:::

Today (2017-10-06), in standard froggy modules, the most recent gcc is version
[4.8.2](https://gcc.gnu.org/gcc-4.8/) (2013) and the most recent python is
3.4.1 (release in May 2014).

We would need to recompile gcc (bad news...) and python. Let's try with
conda...

```bash
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
conda install -c serge-sans-paille gcc_49

conda install ipython
conda install numpy scipy matplotlib
conda install cython

# hdf5 (sequential) and openmpi
module load hdf5/1.8.11_gcc-4.4.6

pip install mpi4py --no-deps --no-binary mpi4py
pip install h5py --no-deps --no-binary h5py

pip install pythran colorlog

# fftw3: we need to recompile fftw3 (.so)...
module load fftw/3.3.4
pip install pyfftw
```

Create the file `~/.pythranrc` with:

```
[pythran]
complex_hook = True

[compiler]
CC=gcc-4.9
CXX=g++-4.9
```
