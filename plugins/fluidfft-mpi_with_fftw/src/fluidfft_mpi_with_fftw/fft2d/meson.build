
py.install_sources(
  '__init__.py',
  subdir: 'fluidfft_mpi_with_fftw/fft2d',
)

pyx = custom_target(
  'mpi_with_fftw1d.pyx',
  output: 'mpi_with_fftw1d.pyx',
  command: ['fluidfft-builder-make-file', '@OUTPUT@', 'FFT2DMPIWithFFTW1D'],
)

pxd = custom_target(
  'fft2dmpi_with_fftw1d.pxd',
  output: 'fft2dmpi_with_fftw1d.pxd',
  command: ['fluidfft-builder-make-file', '@OUTPUT@', 'FFT2DMPIWithFFTW1D'],
)

py.extension_module(
  'mpi_with_fftw1d',
  pyx,
  pxd,
  'fft2dmpi_with_fftw1d.cpp',
  'fft2dmpi_with_fftw1d.h',
  include_path_fluidfft_builder / 'base_fft.cpp',
  include_path_fluidfft_builder / 'base_fft2d.cpp',
  include_path_fluidfft_builder / 'base_fftmpi.cpp',
  include_path_fluidfft_builder / 'base_fft2dmpi.cpp',
  dependencies: dependencies,
  override_options : ['cython_language=cpp'],
  include_directories: include_path_fluidfft_builder,
  install: true,
  subdir: 'fluidfft_mpi_with_fftw/fft2d',
)
