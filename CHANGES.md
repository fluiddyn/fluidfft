# Release notes

See also the [unreleased changes].

## [0.4.4] (2025-07-23)

- New command `fluidfft-get-methods`

## [0.4.3] (2024-11-07)

- Compatibility Python 3.13

## [0.4.2] (2024-08-22)

- Compatibility mpi4py 4.0
- Fix check import classes and `FLUIDFFT_DISABLE_IMPORT_CHECK`.

## [0.4.1] (2024-07-24)

- Support for Numpy 2.0 and Python 3.12

## [0.4.0] (2024-02-11)

- New modular architecture using plugins:

  - Configuration file deprecated.
  - New functions {func}`fluidfft.get_plugins` and {func}`fluidfft.get_methods`.
  - Base C++ code in `fluidfft-builder`
  - [Plugins](#plugins) `f"fluidfft-{s}"` for `s` in `fftw`, `mpi_with_fftw`,
  `fftwmpi`, `pfft` and `p3dfft`.

- Build and upload wheels on PyPI with Github Actions!
- Much better CI in foss.heptapod.net and Github Actions.
- Use the [Meson build system](https://mesonbuild.com) via
  [meson-python](https://github.com/mesonbuild/meson-python).
- Development: use PDM, Nox and Pixi.

## 0.3.5 (2023-09-20)

- Support for Numpy 1.26

## 0.3.4 (2023-01-05)

- Support for Numpy 1.24 and Python 3.11
- Internal: switch to [src layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/)
  and to [Nox](https://nox.thea.codes)

## 0.3.3 (2022-09-13)

- Support for [editable_wheel (PEP 660)](https://peps.python.org/pep-0660/) and setuptools>=65.3
- CI on Github Actions for Linux, Windows and macOS
- Support for Python 3.8, 3.9 and 3.10

## 0.3.2 (2020-10-15)

- Few optimizations in 3D operators
- pyproject.toml and isolated build
- Various bugfixes (in particular, problems with `-Wl,--as-needed`)

## 0.3.1 (2020-05-09)

- Various bugfixes and improvements + compatibility Python 3.8

## 0.3.0 (2019-09-21)

- classes using mpi4py-fft
- compatibility transonic 0.4.0

## 0.2.9 (2019-03-07)

- Minimal support for Windows
- Slightly better logging of bad parameters
- Optimization with dealiasing coeff for fft3dmpi_with_fftw1d

## 0.2.8 (2019-01-27)

- `pip install` from an empty environment should works
- Use transonic
- `compute_spectrum_kykx` (in 2d operators)
- Supports Python >= 3.6

## 0.2.7 (2018-10-22)

- Bugfix setup without mpicxx OpenMPI

## 0.2.5 (2018-07-20)

- Bugfix classes using fftw_mpi (see [issue #14](https://foss.heptapod.net/fluiddyn/fluidfft/issues/14))
- New function mean_global

## 0.2.4 (2018-07-01)

- bugfixes and code improvements
- support PyPy3 and macOS
- More robust to lack of pyfftw and to mkl
- Support fft="default" in operator classes

## 0.2.3 (2018-06-04)

- Less bugs
- Pypy compatibility
- Support jinja2 as fallback for mako
- rotfft_from_vecfft_outin, 3d scatter/gather and compute spectra routines

## 0.2.2

- Less bugs
- Install on clusters
- Better checks for bad input parameters
- Operator div_vv_fft_from_v
- FLUIDDYN_NUM_PROCS_BUILD to install on small computers

## 0.2.1

- install with pip (+ configure install setup with `~/.fluidfft-site.cfg`)

## 0.2.0

- Much cleaner, much less bugs, better unittests
- Much faster building
- Much better operators 2d and 3d
- Tested on different clusters

## 0.1.0

- fluidfft can be used with mpi in fluidsim.

[0.4.0]: https://foss.heptapod.net/fluiddyn/fluidfft/-/compare/0.3.5...0.4.0
[0.4.1]: https://foss.heptapod.net/fluiddyn/fluidfft/-/compare/0.4.0...0.4.1
[0.4.2]: https://foss.heptapod.net/fluiddyn/fluidfft/-/compare/0.4.1...0.4.2
[0.4.3]: https://foss.heptapod.net/fluiddyn/fluidfft/-/compare/0.4.2...0.4.3
[0.4.4]: https://foss.heptapod.net/fluiddyn/fluidfft/-/compare/0.4.3...0.4.4
[unreleased changes]: https://foss.heptapod.net/fluiddyn/fluidfft/-/compare/0.4.4...branch%2Fdefault
