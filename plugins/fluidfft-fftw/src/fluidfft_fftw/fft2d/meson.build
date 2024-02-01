
py.install_sources(
  '__init__.py',
  subdir: 'fluidfft-fftw/ftt2d'
)

with_fftw1d_pyx = custom_target(
  'with_fftw1d.pyx',
  command: ['fluidsim-build-make-pyx', '@OUTPUT@', 'FFT2DWithFFTW1D']
)

with_fftw1d_pxd = custom_target(
  'with_fftw1d.pxd',
  command: ['fluidsim-build-make-pxd', '@OUTPUT@', 'FFT2DWithFFTW1D']
)

# see https://mesonbuild.com/Cython.html
py.extension_module(
  'with_fftw1d',
  pyx,
  dependencies: libfftw3,
  install: true
)

