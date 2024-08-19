(install)=

# Installation and advice

As already written in the overview, Fluidfft is organized as a main package provided few
Fluidfft Python classes using standard packages (Numpy, pyfftw, Dask, etc.) and plugins
which can use other methods, in particular based on C++ classes using more advanced
libraries (as pfft and p3dfft).

In this page we focus on installing the base Fluidfft package for Fluidfft >= 0.4.0.

First, ensure that you have a recent Python installed, since Fluidsim requires Python >=
3.9. Some issues regarding the installation of Python and Python packages are discussed
in
[the main documentation of the project](http://fluiddyn.readthedocs.org/en/latest/install.html).

## Installation with pip

```{note}

We strongly advice to install Fluidfft in a virtual environment. See the
official guide [Install packages in a virtual environment using pip and
venv](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/).

```

Fluidfft can be installed without compilation with `pip`:

```sh
pip install pip -U
pip install fluidfft
```

However, fluidfft [works with plugins](./plugins.md), which have to be installed
additionally, for example using fluidfft optional dependencies. Installing [pyfftw] does
not require compilation:

```sh
pip install "fluidfft[pyfftw]"
```

whereas installing other plugins will trigger local compilation.

```sh
pip install "fluidfft[fftw,mpi-with-fftw,fftwmpi]"
```

## Installation with conda

```sh
conda install fluidfft
```

### Remark on Numpy installed with conda

In anaconda (or miniconda), Numpy installed with `conda install numpy` can be built and
linked with MKL (an Intel library). This can be a real plus for performance since MKL
replaces fftw functions by (usually) faster ones but it has a drawback for fft using the
library fftw3_mpi (an implementation of parallel fft using 1D decomposition by fftw). MKL
implements some fftw functions but not all the functions defined in fftw3_mpi. Since the
libraries are loaded dynamically, if numpy is imported before the fftw_mpi libraries,
this can lead to very bad issues (segmentation fault, only if numpy is imported before
the class!).

To install with anaconda numpy linked with openblas:

```
conda config --add channels conda-forge
conda install "blas[build=*openblas]" numpy
```

## Environment variables

Fluidfft is sensible at runtime to the environment variable `TRANSONIC_BACKEND`. The
Transonic backend is "pythran" by default, but it can also be set to "python" or "numba".

[pyfftw]: https://github.com/pyFFTW/pyFFTW/
