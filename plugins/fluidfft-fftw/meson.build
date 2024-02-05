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

py_mod = import('python')
py = py_mod.find_installation('python3', pure: false)
py_dep = py.dependency()

incdir_numpy = run_command(
  'transonic-get-include', 'numpy', check: true
).stdout().strip()
inc_np = include_directories(incdir_numpy)
np_dep = declare_dependency(include_directories: inc_np)

fftw_dep = dependency('fftw3', static: false)

dependencies = [fftw_dep, np_dep]

include_path_fluidfft_builder = run_command(
  'fluidfft-builder-print-include-dir', check: true
).stdout().strip()

include_path_cy = run_command(
  'fluidfft-builder-print-include-dir-cython', check: true
).stdout().strip()
add_project_arguments('-I', include_path_cy, language : 'cython')

subdir('src/fluidfft_fftw')
