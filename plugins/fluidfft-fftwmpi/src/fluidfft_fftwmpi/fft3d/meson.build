
py.install_sources(
  '__init__.py',
  subdir: 'fluidfft_fftwmpi/fft3d',
)

pyx = custom_target(
  'mpi_with_fftwmpi3d.pyx',
  output: 'mpi_with_fftwmpi3d.pyx',
  command: ['fluidfft-builder-make-file', '@OUTPUT@', 'FFT3DMPIWithFFTWMPI3D'],
)

pxd = custom_target(
  'fft3dmpi_with_fftwmpi3d.pxd',
  output: 'fft3dmpi_with_fftwmpi3d.pxd',
  command: ['fluidfft-builder-make-file', '@OUTPUT@', 'FFT3DMPIWithFFTWMPI3D'],
)

py.extension_module(
  'mpi_with_fftwmpi3d',
  pyx,
  pxd,
  'fft3dmpi_with_fftwmpi3d.cpp',
  'fft3dmpi_with_fftwmpi3d.h',
  include_path_fluidfft_builder / 'base_fft.cpp',
  include_path_fluidfft_builder / 'base_fft3d.cpp',
  include_path_fluidfft_builder / 'base_fftmpi.cpp',
  include_path_fluidfft_builder / 'base_fft3dmpi.cpp',
  dependencies: dependencies,
  override_options : ['cython_language=cpp'],
  include_directories: include_path_fluidfft_builder,
  install: true,
  subdir: 'fluidfft_fftwmpi/fft3d',
  link_args: link_args,
)
