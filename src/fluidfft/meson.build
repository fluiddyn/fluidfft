python_sources = [
  '__init__.py',
  '_version.py',
  'base.py',
  'bench.py',
  'bench_analysis.py',
  'cli_get_methods.py',
  'util.py',
  'build_conf.txt',
]

run_command(
  'sh',
  '-c',
  'echo transonic_backend=' + backend + ' > ' + 'build_conf.txt',
  check: true,
)


py.install_sources(
  python_sources,
  subdir: 'fluidfft'
)

subdir('fft2d')
subdir('fft3d')
