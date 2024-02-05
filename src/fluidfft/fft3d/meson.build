python_sources = [
  '__init__.py',
  'base.py',
  'operators.py',
  'testing.py',
  'with_pyfftw.py',
]

fs = import('fs')
if fs.is_file('fake_mod_fft3d_for_doc.py')
  python_sources += ['fake_mod_fft3d_for_doc.py']
endif

py.install_sources(
  python_sources,
  subdir: 'fluidfft/fft3d'
)

run_command(['transonic', '--meson', '--backend', backend, 'operators.py'], check: true)

foreach be : backends
  subdir('__' + be + '__')
endforeach
