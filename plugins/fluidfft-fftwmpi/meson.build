project(
  'fluidfft-fftwmpi',
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

fftw_dep = dependency('fftw3', static: false)
mpi_dep = dependency('mpi', language: 'cpp')
# no fftw3-mpi.pc file, see https://github.com/FFTW/fftw3/issues/57
# fftwmpi_dep = dependency('fftw3-mpi', static: false)

compiler = meson.get_compiler('cpp')
# fftw_mpi is not found on Ubuntu
fftwmpi_dep = compiler.find_library('fftw_mpi', required: false)

py_mod = import('python')
py = py_mod.find_installation('python3', pure: false)
py_dep = py.dependency()

incdir_numpy = run_command(
  'transonic-get-include', 'numpy', check: true
).stdout().strip()
inc_np = include_directories(incdir_numpy)
np_dep = declare_dependency(include_directories: inc_np)

dependencies = [fftw_dep, mpi_dep, fftwmpi_dep, np_dep]
link_args = ['-lfftw3_mpi']

include_path_fluidfft_builder = run_command(
  'fluidfft-builder-print-include-dir', check: true
).stdout().strip()

include_path_cy = run_command(
  'fluidfft-builder-print-include-dir-cython', check: true
).stdout().strip()
add_project_arguments('-I', include_path_cy, language : 'cython')

subdir('src/fluidfft_fftwmpi')
