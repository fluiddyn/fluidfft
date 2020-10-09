from runpy import run_path
from pathlib import Path
import sys

from setuptools import setup, find_packages, Extension

here = Path(__file__).parent.absolute()
sys.path.insert(0, ".")

from setup_configure import (
    TRANSONIC_BACKEND,
    build_needs_mpi4py,
    build_dependencies_backends,
)
from setup_build import FluidFFTBuildExt

# Get the long description from the relevant file
with open("README.rst") as file:
    lines = file.readlines()
for i, line in enumerate(lines):
    if line.endswith(":alt: Code coverage\n"):
        iline_coverage = i
        break
long_description = "".join(lines[iline_coverage + 2 :])

# Get the version from the relevant file
d = run_path("fluidfft/_version.py")
__version__ = d["__version__"]

# Set entry points
entry_points = {
    "console_scripts": [
        "fluidfft-bench = fluidfft.bench:run",
        "fluidfft-bench-analysis = fluidfft.bench_analysis:run",
    ]
}

# Set setup_requires and install_requires depending on the configuration
install_requires = ["fluiddyn >= 0.2.3", "transonic >= 0.4"]
setup_requires = []
setup_requires.extend(build_dependencies_backends[TRANSONIC_BACKEND])
if build_needs_mpi4py:
    setup_requires.append("mpi4py")
    install_requires.append("mpi4py")

setup(
    version=__version__,
    long_description=long_description,
    packages=find_packages(
        exclude=["doc", "include", "scripts", "src_cpp", "src_cy"]
    ),
    entry_points=entry_points,
    setup_requires=setup_requires,
    install_requires=install_requires,
    # To trick build into running build_ext (taken from h5py)
    ext_modules=[Extension("trick.x", ["trick.c"])],
    cmdclass={"build_ext": FluidFFTBuildExt},
)
