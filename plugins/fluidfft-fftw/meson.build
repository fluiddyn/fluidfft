project(
  'fluidfft-fftw',
  'cpp',
  license: 'CeCILL',
  meson_version: '>= 1.1.0',
  default_options: [
    'buildtype=release',
    'c_std=c99',
    'cpp_std=c++14',
  ],
)

# https://mesonbuild.com/Python-module.html
py_mod = import('python')
py = py_mod.find_installation('python3', pure: false)
py_dep = py.dependency()

subdir('src/fluidfft_fftw')
