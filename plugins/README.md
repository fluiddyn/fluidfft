# Fluidfft plugins

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

The following plugins are implemented in Fluidfft repository:

- [x] fluidfft-fftw
- [x] fluidfft-mpi_with_fftw (parallel methods using the sequential FFTW3 library)
- [x] fluidfft-fftwmpi (methods using the library `libfftw_mpi`)
- [x] fluidfft-p3dfft
- [x] fluidfft-pfft
- [x] fluidfft-mpi4pyfft (cannot be tested because mpi4py-fft installation fails)

We plan to soon also have:

- [ ] fluidfft-pyvkfft (https://pyvkfft.readthedocs.io)
