project(
  'fluidfft-hipfft',
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

# HIP/ROCm dependencies
hip_dep = dependency('hip', required: true)
hipfft_dep = dependency('hipfft', required: true)

dependencies = [hip_dep, hipfft_dep, np_dep]

include_path_fluidfft_builder = run_command(
  'fluidfft-builder-print-include-dir', check: true
).stdout().strip()
add_project_arguments('-I', include_path_fluidfft_builder, language : 'cpp')
# Workaround for __builtin_cpu_init() not being supported on GPU targets
add_project_arguments('-D__builtin_cpu_init()=', language : 'cpp')


include_path_cy = run_command(
  'fluidfft-builder-print-include-dir-cython', check: true
).stdout().strip()
add_project_arguments('-I', include_path_cy, language : 'cython')

subdir('src/fluidfft_hipfft')
