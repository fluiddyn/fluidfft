project(
  'fluidfft-p3dfft',
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

P3DFFT_DIR = run_command(
  py, '-c', 'import os; print(os.environ.get("P3DFFT_DIR", ""))',
  check: true
).stdout().strip()
message('P3DFFT_DIR=' + P3DFFT_DIR)

if P3DFFT_DIR != ''
  P3DFFT_LIB_DIR = P3DFFT_DIR + '/lib'
  P3DFFT_INCLUDE_DIR = P3DFFT_DIR + '/include'
else
  P3DFFT_LIB_DIR = run_command(
    py, '-c', 'import os; print(os.environ.get("P3DFFT_LIB_DIR", ""))',
    check: true
  ).stdout().strip()
  message('P3DFFT_LIB_DIR=' + P3DFFT_LIB_DIR)

  P3DFFT_INCLUDE_DIR = run_command(
    py, '-c', 'import os; print(os.environ.get("P3DFFT_INCLUDE_DIR", ""))',
    check: true
  ).stdout().strip()
  message('P3DFFT_INCLUDE_DIR=' + P3DFFT_INCLUDE_DIR)
endif

dirs = []
if P3DFFT_LIB_DIR != ''
   dirs += [P3DFFT_LIB_DIR]
endif

include_directories = []
if P3DFFT_INCLUDE_DIR != ''
  include_directories = include_directories(P3DFFT_INCLUDE_DIR)
endif

p3dfft_dep = compiler.find_library(
  'p3dfft',
  dirs: dirs,
  has_headers: ['p3dfft.h'],
  header_include_directories: include_directories,
  required: true,
)
link_args = ['-lfftw3_mpi']

dependencies = [fftw_dep, mpi_dep, np_dep, fftwmpi_dep, p3dfft_dep]

include_path_fluidfft_builder = run_command(
  'fluidfft-builder-print-include-dir', check: true
).stdout().strip()

include_directories = [include_directories, include_path_fluidfft_builder]

include_path_cy = run_command(
  'fluidfft-builder-print-include-dir-cython', check: true
).stdout().strip()
add_project_arguments('-I', include_path_cy, language : 'cython')

subdir('src/fluidfft_p3dfft')
