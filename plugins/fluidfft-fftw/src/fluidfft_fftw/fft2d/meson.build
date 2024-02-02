
py.install_sources(
  '__init__.py',
  subdir: 'fluidfft_fftw/fft2d',
)

pyx = custom_target(
  'with_fftw1d.pyx',
  output: 'with_fftw1d.pyx',
  command: ['fluidfft-builder-make-file', '@OUTPUT@', 'FFT2DWithFFTW1D'],
)

pxd = custom_target(
  'fft2d_with_fftw1d.pxd',
  output: 'fft2d_with_fftw1d.pxd',
  command: ['fluidfft-builder-make-file', '@OUTPUT@', 'FFT2DWithFFTW1D'],
)

py.extension_module(
  'with_fftw1d',
  pyx,
  pxd,
  'fft2d_with_fftw1d.cpp',
  'fft2d_with_fftw1d.h',
  include_path_fluidfft_builder / 'base_fft2d.cpp',
  include_path_fluidfft_builder / 'base_fft.cpp',
  dependencies: fftw_dep,
  override_options : ['cython_language=cpp'],
  include_directories: include_path_fluidfft_builder,
  install: true,
  subdir: 'fluidfft_fftw/fft2d',
)
