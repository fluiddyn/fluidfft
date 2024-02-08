
py.install_sources('__init__.py', subdir: 'fluidfft_p3dfft')

pyx = custom_target(
  'mpi_with_p3dfft.pyx',
  output: 'mpi_with_p3dfft.pyx',
  command: ['fluidfft-builder-make-file', '@OUTPUT@', 'FFT3DMPIWithP3DFFT'],
)

pxd = custom_target(
  'fft3dmpi_with_p3dfft.pxd',
  output: 'fft3dmpi_with_p3dfft.pxd',
  command: ['fluidfft-builder-make-file', '@OUTPUT@', 'FFT3DMPIWithP3DFFT'],
)

py.extension_module(
  'mpi_with_p3dfft',
  pyx,
  pxd,
  'fft3dmpi_with_p3dfft.cpp',
  'fft3dmpi_with_p3dfft.h',
  include_path_fluidfft_builder / 'base_fft.cpp',
  include_path_fluidfft_builder / 'base_fft3d.cpp',
  include_path_fluidfft_builder / 'base_fftmpi.cpp',
  include_path_fluidfft_builder / 'base_fft3dmpi.cpp',
  dependencies: dependencies,
  override_options : ['cython_language=cpp'],
  include_directories: include_directories,
  link_args: link_args,
  install: true,
  subdir: 'fluidfft_p3dfft',
)
