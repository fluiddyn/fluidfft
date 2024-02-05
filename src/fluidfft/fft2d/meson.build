python_sources = [
  '__init__.py',
  'operators.py',
  'with_pyfftw.py',
  'with_dask.py',
  'testing.py',
]

fs = import('fs')
if fs.is_file('fake_mod_fft2d_for_doc.py')
  python_sources += ['fake_mod_fft2d_for_doc.py']
endif

py.install_sources(
  python_sources,
  subdir: 'fluidfft/fft2d'
)

run_command(['transonic', '--meson', '--backend', backend, 'operators.py'], check: true)

foreach be : backends
  subdir('__' + be + '__')
endforeach
