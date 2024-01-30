python_sources = [
  '__init__.py',
  'operators.py',
  'with_dask.py',
  'testing.py',
]

py.install_sources(
  python_sources,
  subdir: 'fluidfft/fft2d'
)

run_command(['transonic', '--meson', '--backend', backend, 'operators.py'], check: true)

foreach be : backends
  subdir('__' + be + '__')
endforeach
