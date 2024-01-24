# Installation on the HPC cluster Triolith (NSC)

`Triolith is a cluster of the Swedish National Infrastructure for Computing
(SNIC) <https://www.nsc.liu.se/systems/triolith/>`.

First install mercurial in python 2.

```bash
module load gcc/6.2.0
module load python/2.7.12
pip2 install mercurial --user
```

Load necessary modules

```bash
module load python3/3.6.1
module load openmpi/1.6.5-g44
module load hdf5/1.8.11-i1214-parallel
module load libtool/2.4
module load autoconf/2.69
```

To build and install in FFTW, P3DFFT and PFFT, use the scripts provided in this
page:

> ```{toctree}
> :maxdepth: 1
>
> fft_libs
> ```

but with few modifications:

> - P3DFFT
>
>   - Use `autoreconf -fvi` just before `./configure ...` and `make`.
>   - Use `make -i install` to finish installation, while making a note of
>     the errors encountered. Ignore if only P3DFFT samples fail to build.

Setup a virtual environment, using `virtualenv`, or `mkvirtualenv` command
from `virtualenvwrapper` package or simply using Python's built-in module
`python -m venv` module.

Create the file `~/.pythranrc` with:

```
[pythran]
complex_hook = True

[compiler]
cflags=-std=c++11 -fno-math-errno -w -fwhole-program -fvisibility=hidden -I$NSC_COMP_BIN_PATH/../include/c++/$NSC_COMP_VER
ldflags=-fvisibility=hidden -Wl,-strip-all -L$NSC_COMP_LIB_PATH
```

Set environment variable `LD_LIBRARY_PATH` as:

```
export LD_LIBRARY_PATH=$HOME/.local/lib:$NSC_COMP_LIB_PATH:$LD_LIBRARY_PATH
```

Set up `~/.fluidfft-site.cfg` and install python packages as described in
occigen installation:

> ```{toctree}
> :maxdepth: 1
>
> occigen
> ```
