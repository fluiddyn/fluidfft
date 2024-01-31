python_sources = [
  '__init__.py',
]

py.install_sources(
  python_sources,
  subdir: 'fluidfft-fftw/ftt2d'
)

run_command(
  py,
  [
    '-c',
    '''from fluidfft_build_deps.src_cy.make_files_with_mako import make_file
make_file("fft2d_with_fftw1d", "FFT2DWithFFTW1D", "numpy")'''
  ],
  check: true
)

src_cpp_path = run_command(
  py,
  [
    '-c',
    '''import fluidfft_build_deps.src_cpp
print(fluidfft_build_deps.src_cpp.__path__[0])'''
  ],
  check: true
).stdout().strip()

base_sources = ['base_fft.h']
foreach filename : base_sources
  src = join_paths(src_cpp_path, 'base', filename)
  dest = join_paths(meson.current_source_dir(), filename)
  run_command('cp', src, dest, check : true)
endforeach

sources = ['base_fft2d.h', 'fft2d_with_fftw1d.cpp', 'fft2d_with_fftw1d.h']
foreach filename : sources
  src = join_paths(src_cpp_path, '2d', filename)
  dest = join_paths(meson.current_source_dir(), filename)
  run_command('cp', src, dest, check : true)
endforeach

# shared_library(
#   'fft2d_with_fftw1d',
#   ['fft2d_with_fftw1d.cpp', 'fft2d_with_fftw1d.h'],
#   install : true,
# )
