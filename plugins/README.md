# Fluidfft plugins and methods

The main Fluidfft package only contains pure Python FFT classes using other
packages to perform the transforms.

The classes using in the background C++ Fluidfft classes are implemented in
Fluidfft plugins. A plugin is a small Python packages defining entry points
`fluidfft.plugins`.

For example, the plugin `fluidfft-fftw` defines 3 sequential methods declared
in its `pyproject.toml` file like this:

```toml
[project.entry-points."fluidfft.plugins"]

"fft2d.with_fftw1d" = "fluidfft_fftw.fft2d.with_fftw1d"
"fft2d.with_fftw2d" = "fluidfft_fftw.fft2d.with_fftw2d"
"fft3d.with_fftw3d" = "fluidfft_fftw.fft3d.with_fftw3d"
```

The method strings (for example `"fft3d.with_fftw3d"`) can be given to
{func}`fluidfft.import_fft_class` and Operator classes (like
{class}`fluidfft.fft3d.operators.OperatorsPseudoSpectral3D`).

The methods available can be obtained with the command line `fluidfft-get-methods`
and the function {func}`fluidfft.get_methods`.

The following plugins are implemented in Fluidfft repository:

- `fluidfft-fftw`
- `fluidfft-mpi_with_fftw` (parallel methods using the sequential FFTW3 library)
- `fluidfft-fftwmpi` (methods using the library `libfftw_mpi`)
- `fluidfft-p3dfft`
- `fluidfft-pfft`
- `fluidfft-mpi4pyfft` (cannot be tested because mpi4py-fft installation fails)

We plan to soon also have:

- `fluidfft-pyvkfft` (<https://pyvkfft.readthedocs.io>)

There are few plugins available on PyPI (see <https://pypi.org/search/?q=fluidfft>).
Note that their installation can trigger compilation (see [](#build-from-source))
and that the corresponding library has to be installed first.

## Install FFT libraries

Here is a list of FFT libraries, with instructions on
how to install them:

```{toctree}
:maxdepth: 1

install/fft_libs
```
