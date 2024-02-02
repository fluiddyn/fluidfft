python_sources = [
  '__init__.py',
  'base.py',
  'operators.py',
  'testing.py',
  'with_pyfftw.py',
]

py.install_sources(
  python_sources,
  subdir: 'fluidfft/fft3d'
)

run_command(['transonic', '--meson', '--backend', backend, 'operators.py'], check: true)

foreach be : backends
  subdir('__' + be + '__')
endforeach
