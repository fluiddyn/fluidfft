
py.install_sources(
  '__init__.py',
  subdir: 'fluidfft_fftwmpi/fft2d',
)

pyx = custom_target(
  'mpi_with_fftwmpi2d.pyx',
  output: 'mpi_with_fftwmpi2d.pyx',
  command: ['fluidfft-builder-make-file', '@OUTPUT@', 'FFT2DMPIWithFFTWMPI2D'],
)

pxd = custom_target(
  'fft2dmpi_with_fftwmpi2d.pxd',
  output: 'fft2dmpi_with_fftwmpi2d.pxd',
  command: ['fluidfft-builder-make-file', '@OUTPUT@', 'FFT2DMPIWithFFTWMPI2D'],
)

py.extension_module(
  'mpi_with_fftwmpi2d',
  pyx,
  pxd,
  'fft2dmpi_with_fftwmpi2d.cpp',
  'fft2dmpi_with_fftwmpi2d.h',
  include_path_fluidfft_builder / 'base_fft.cpp',
  include_path_fluidfft_builder / 'base_fft2d.cpp',
  include_path_fluidfft_builder / 'base_fftmpi.cpp',
  include_path_fluidfft_builder / 'base_fft2dmpi.cpp',
  dependencies: dependencies,
  override_options : ['cython_language=cpp'],
  include_directories: include_path_fluidfft_builder,
  install: true,
  subdir: 'fluidfft_fftwmpi/fft2d',
  link_args: link_args,
)
