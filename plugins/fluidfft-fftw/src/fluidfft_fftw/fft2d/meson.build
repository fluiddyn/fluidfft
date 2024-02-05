
py.install_sources(
  '__init__.py',
  subdir: 'fluidfft_fftw/fft2d',
)

foreach dim : ['1', '2']

  pyx = custom_target(
    f'with_fftw@dim@d.pyx',
    output: f'with_fftw@dim@d.pyx',
    command: ['fluidfft-builder-make-file', '@OUTPUT@', f'FFT2DWithFFTW@dim@D'],
  )

  pxd = custom_target(
    f'fft2d_with_fftw@dim@d.pxd',
    output: f'fft2d_with_fftw@dim@d.pxd',
    command: ['fluidfft-builder-make-file', '@OUTPUT@', f'FFT2DWithFFTW@dim@D'],
  )

  py.extension_module(
    f'with_fftw@dim@d',
    pyx,
    pxd,
    f'fft2d_with_fftw@dim@d.cpp',
    f'fft2d_with_fftw@dim@d.h',
    include_path_fluidfft_builder / 'base_fft2d.cpp',
    include_path_fluidfft_builder / 'base_fft.cpp',
    dependencies: dependencies,
    override_options : ['cython_language=cpp'],
    include_directories: include_path_fluidfft_builder,
    install: true,
    subdir: 'fluidfft_fftw/fft2d',
  )

endforeach
