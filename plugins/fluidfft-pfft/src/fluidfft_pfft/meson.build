
py.install_sources('__init__.py', subdir: 'fluidfft_pfft')

pyx = custom_target(
  'mpi_with_pfft.pyx',
  output: 'mpi_with_pfft.pyx',
  command: ['fluidfft-builder-make-file', '@OUTPUT@', 'FFT3DMPIWithPFFT'],
)

pxd = custom_target(
  'fft3dmpi_with_pfft.pxd',
  output: 'fft3dmpi_with_pfft.pxd',
  command: ['fluidfft-builder-make-file', '@OUTPUT@', 'FFT3DMPIWithPFFT'],
)

py.extension_module(
  'mpi_with_pfft',
  pyx,
  pxd,
  'fft3dmpi_with_pfft.cpp',
  'fft3dmpi_with_pfft.h',
  include_path_fluidfft_builder / 'base_fft.cpp',
  include_path_fluidfft_builder / 'base_fft3d.cpp',
  include_path_fluidfft_builder / 'base_fftmpi.cpp',
  include_path_fluidfft_builder / 'base_fft3dmpi.cpp',
  dependencies: dependencies,
  override_options : ['cython_language=cpp'],
  include_directories: include_directories,
  link_args: link_args,
  install: true,
  subdir: 'fluidfft_pfft',
)
