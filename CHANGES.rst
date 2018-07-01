0.2.4 (2018-07-01)
------------------

- bugfixes and code improvements
- support PyPy3 and macOS
- More robust to lack of pyfftw and to mkl
- Support fft="default" in operator classes

0.2.3 (2018-06-04)
------------------

- Less bugs
- Pypy compatibility
- Support jinja2 as fallback for mako
- rotfft_from_vecfft_outin, 3d scatter/gather and compute spectra routines

0.2.2
-----

- Less bugs
- Install on clusters
- Better checks for bad input parameters
- Operator div_vv_fft_from_v
- FLUIDDYN_NUM_PROCS_BUILD to install on small computers

0.2.1
-----

- install with pip (+ configure install setup with ``~/.fluidfft-site.cfg``)

0.2.0
-----

- Much cleaner, much less bugs, better unittests
- Much faster building
- Much better operators 2d and 3d
- Tested on different clusters

0.1.0
-----

- fluidfft can be used with mpi in fluidsim.
