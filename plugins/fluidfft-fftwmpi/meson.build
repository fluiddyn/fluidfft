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

py_mod = import('python')
py = py_mod.find_installation('python3', pure: false)
py_dep = py.dependency()

fftw_dep = dependency('fftw3', static: false)
mpi_dep = dependency('mpi', language: 'cpp')
# no fftw3-mpi.pc file, see https://github.com/FFTW/fftw3/issues/57
# fftwmpi_dep = dependency('fftw3-mpi', static: false)

compiler = meson.get_compiler('cpp')
fftwmpi_dep = compiler.find_library('libfftw_mpi', required: true)

dependencies = [fftw_dep, mpi_dep, fftwmpi_dep]
link_args = ['-lfftw3_mpi']

include_path_fluidfft_builder = run_command('fluidfft-builder-print-include-dir').stdout().strip()

include_path_cy = run_command('fluidfft-builder-print-include-dir-cython').stdout().strip()
add_project_arguments('-I', include_path_cy, language : 'cython')

subdir('src/fluidfft_fftwmpi')
