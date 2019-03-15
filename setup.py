import os
from runpy import run_path
from pathlib import Path

from setuptools import setup, find_packages, Extension

here = Path(__file__).parent.absolute()

try:
    from setup_build import FluidFFTBuildExt
except ImportError:
    # fix a bug... Useful when there is already a module with the same name
    # imported.
    FluidFFTBuildExt = run_path(here / "setup_build.py")["FluidFFTBuildExt"]


# Bootstrapping dependencies required for the setup
setup_requires = ["numpy", "cython", "jinja2", "transonic>=0.2.0"]
on_tox = os.getenv("TOXENV")
if on_tox is not None:
    setup_requires.append("mpi4py")

# Get the long description from the relevant file
with open("README.rst") as f:
    long_description = f.read()
lines = long_description.splitlines(True)
for i, line in enumerate(lines):
    if line.endswith(":alt: Code coverage\n"):
        iline_coverage = i
        break

long_description = "".join(lines[iline_coverage + 2 :])

# Get the version from the relevant file
d = run_path("fluidfft/_version.py")
__version__ = d["__version__"]

entry_points = {
    "console_scripts": [
        "fluidfft-bench = fluidfft.bench:run",
        "fluidfft-bench-analysis = fluidfft.bench_analysis:run",
    ]
}


setup(
    version=__version__,
    long_description=long_description,
    packages=find_packages(
        exclude=["doc", "include", "scripts", "src_cpp", "src_cy"]
    ),
    entry_points=entry_points,
    setup_requires=setup_requires,
    # To trick build into running build_ext (taken from h5py)
    ext_modules=[Extension("trick.x", ["trick.c"])],
    cmdclass={"build_ext": FluidFFTBuildExt},
)
