# Fluidfft plugins

Directory containing the plugins, i.e. Python packages declaring the
`fluidfft.plugins` entry point.

We should have

- [x] fluidfft-mpi4pyfft (cannot be tested because mpi4py-fft installation fails)
- [ ] fluidfft-fftw
- [ ] fluidfft-mpi_with_fftw
- [ ] fluidfft-fftwmpi
- [ ] fluidfft-p3dfft
- [ ] fluidfft-pfft
- [ ] fluidfft-pyvkfft (https://pyvkfft.readthedocs.io)

Currently, we have only one tested plugin (fluidfft-pyfftw), which was written to
design and test the plugin machinery. However, I (PA) think that this (pure Python)
code will have to go back in fluidfft. Pyfftw can just be an optional dependency
for fluidfft.

TODO: When we have other plugins, move back the code using pyfftw inside fluidfft.
