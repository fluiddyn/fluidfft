
py.install_sources(
  '__init__.py',
  subdir: 'fluidfft_hipfft/fft2d',
)

pyx = custom_target(
  'with_hipfft.pyx',
  output: 'with_hipfft.pyx',
  command: ['fluidfft-builder-make-file', '@OUTPUT@', 'FFT2DWithHIPFFT'],
)

pxd = custom_target(
  'fft2d_with_hipfft.pxd',
  output: 'fft2d_with_hipfft.pxd',
  command: ['fluidfft-builder-make-file', '@OUTPUT@', 'FFT2DWithHIPFFT'],
)

py.extension_module(
  'with_hipfft',
  pyx,
  pxd,
  'fft2d_with_hipfft.cpp',
  'fft2d_with_hipfft.h',
  include_path_fluidfft_builder / 'base_fft2d.cpp',
  include_path_fluidfft_builder / 'base_fft.cpp',
  dependencies: dependencies,
  override_options : ['cython_language=cpp'],
  include_directories: include_path_fluidfft_builder,
  install: true,
  subdir: 'fluidfft_hipfft/fft2d',
)
