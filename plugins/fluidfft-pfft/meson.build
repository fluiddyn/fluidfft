project(
  'fluidfft-pfft',
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

fftw_dep = dependency('fftw3', static: false)
mpi_dep = dependency('mpi', language: 'cpp')

incdir_numpy = run_command(
  'transonic-get-include', 'numpy', check: true
).stdout().strip()
inc_np = include_directories(incdir_numpy)
np_dep = declare_dependency(include_directories: inc_np)

compiler = meson.get_compiler('cpp')
# fftw_mpi is not found on Ubuntu
fftwmpi_dep = compiler.find_library('fftw_mpi', required: false)


PFFT_DIR = run_command(
  py, '-c', 'import os; print(os.environ.get("PFFT_DIR", ""))',
  check: true
).stdout().strip()
message('PFFT_DIR=' + PFFT_DIR)

PFFT_LIB_DIR = run_command(
  py, '-c', 'import os; print(os.environ.get("PFFT_LIB_DIR", ""))',
  check: true
).stdout().strip()
message('PFFT_LIB_DIR=' + PFFT_LIB_DIR)

PFFT_INCLUDE_DIR = run_command(
  py, '-c', 'import os; print(os.environ.get("PFFT_INCLUDE_DIR", ""))',
  check: true
).stdout().strip()
message('PFFT_INCLUDE_DIR=' + PFFT_INCLUDE_DIR)

# We need to do something with these values
# see https://stackoverflow.com/a/61072768/1779806

# set LIBRARY_PATH to include the directory where is libpfft.so
pfft_dep = compiler.find_library('pfft', required: true)
link_args = ['-lpfft', '-lfftw3_mpi']

dependencies = [fftw_dep, mpi_dep, np_dep, fftwmpi_dep, pfft_dep]

include_path_fluidfft_builder = run_command(
  'fluidfft-builder-print-include-dir', check: true
).stdout().strip()

include_path_cy = run_command(
  'fluidfft-builder-print-include-dir-cython', check: true
).stdout().strip()
add_project_arguments('-I', include_path_cy, language : 'cython')

subdir('src/fluidfft_pfft')
