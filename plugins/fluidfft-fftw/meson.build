project(
  'fluidfft-fftw',
  'cpp',
  'cython',
  license: 'CeCILL',
  meson_version: '>= 1.1.0',
  default_options: [
    'buildtype=release',
    'c_std=c99',
    'cpp_std=c++11',
  ],
)

# https://mesonbuild.com/Python-module.html
py_mod = import('python')
py = py_mod.find_installation('python3', pure: false)
py_dep = py.dependency()

libfftw3 = dependency('libfftw3', static: false)

include_path_fluidfft_builder = run_command("fluidsim-build-get-include-path")

subdir('src/fluidfft_fftw')
