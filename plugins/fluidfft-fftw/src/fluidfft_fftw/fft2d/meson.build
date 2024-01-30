python_sources = [
  '__init__.py',
]

py.install_sources(
  python_sources,
  subdir: 'fluidfft-fftw/ftt2d'
)

cpp_path = run_command(
  py,
  [
    '-c',
    '''import fluidfft_build_deps.src_cpp
print(fluidfft_build_deps.src_cpp.__path__[0])'''
  ],
  check: true
).stdout().strip()


run_command(['transonic', '--meson', '--backend', backend, 'operators2d.py', 'operators3d.py'], check: true)

foreach be : backends
  subdir('__' + be + '__')
endforeach
