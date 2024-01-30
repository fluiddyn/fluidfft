from pathlib import Path
import sys

from setuptools import setup, Extension

here = Path(__file__).parent.absolute()
sys.path.insert(0, ".")

from setup_configure import (
    TRANSONIC_BACKEND,
    build_needs_mpi4py,
    build_dependencies_backends,
)
from setup_build import FluidFFTBuildExt

# Set setup_requires depending on the configuration
setup_requires = []
setup_requires.extend(build_dependencies_backends[TRANSONIC_BACKEND])
if build_needs_mpi4py:
    setup_requires.append("mpi4py")

setup(
    setup_requires=setup_requires,
    # To trick build into running build_ext (taken from h5py)
    ext_modules=[Extension("trick.x", ["trick.c"])],
    cmdclass={"build_ext": FluidFFTBuildExt},
)
