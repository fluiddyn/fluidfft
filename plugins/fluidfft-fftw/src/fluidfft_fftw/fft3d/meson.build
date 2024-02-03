
py.install_sources(
  '__init__.py',
  subdir: 'fluidfft_fftw/fft3d',
)

pyx = custom_target(
  'with_fftw3d.pyx',
  output: 'with_fftw3d.pyx',
  command: ['fluidfft-builder-make-file', '@OUTPUT@', 'FFT3DWithFFTW3D'],
)

pxd = custom_target(
  'fft3d_with_fftw3d.pxd',
  output: 'fft3d_with_fftw3d.pxd',
  command: ['fluidfft-builder-make-file', '@OUTPUT@', 'FFT3DWithFFTW3D'],
)

py.extension_module(
  'with_fftw3d',
  pyx,
  pxd,
  'fft3d_with_fftw3d.cpp',
  'fft3d_with_fftw3d.h',
  include_path_fluidfft_builder / 'base_fft3d.cpp',
  include_path_fluidfft_builder / 'base_fft.cpp',
  dependencies: dependencies,
  override_options : ['cython_language=cpp'],
  include_directories: include_path_fluidfft_builder,
  install: true,
  subdir: 'fluidfft_fftw/fft3d',
)
