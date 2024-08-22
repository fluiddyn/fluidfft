"""Efficient and easy Fast Fourier Transform for Python
=======================================================

The fft and related `operators` classes are in the two subpackages

.. autosummary::
   :toctree:

   fft2d
   fft3d

The two commands ``fluidfft-bench`` and ``fluidfft-bench-analysis`` can be used to
benchmark the classes on particular cases and computers. These commands are
implemented in the following modules

.. autosummary::
   :toctree:

   bench
   bench_analysis

This root module provides two helper functions to import fft classes and create
fft objects:

.. autofunction:: get_plugins

.. autofunction:: get_methods

.. autofunction:: import_fft_class

.. autofunction:: create_fft_object

"""

import importlib
import os
import subprocess
import sys
import logging

try:
    from importlib_metadata import entry_points, EntryPoint
except ImportError:
    # for sys.version_info >= (3, 10), no need for importlib_metadata
    from importlib.metadata import entry_points, EntryPoint

from fluiddyn.util import mpi

from fluidfft._version import __version__

try:
    from pyfftw import empty_aligned, byte_align
except ImportError:
    import numpy as np

    empty_aligned = np.empty

    def byte_align(values, *args):
        """False byte_align function used when pyfftw can not be imported"""
        return values


__citation__ = r"""
@article{fluiddyn,
doi = {10.5334/jors.237},
year = {2019},
publisher = {Ubiquity Press,  Ltd.},
volume = {7},
author = {Pierre Augier and Ashwin Vishnu Mohanan and Cyrille Bonamy},
title = {{FluidDyn}: A Python Open-Source Framework for Research and Teaching in Fluid Dynamics
    by Simulations,  Experiments and Data Processing},
journal = {Journal of Open Research Software}
}

@article{fluidfft,
doi = {10.5334/jors.238},
year = {2019},
publisher = {Ubiquity Press,  Ltd.},
volume = {7},
author = {Ashwin Vishnu Mohanan and Cyrille Bonamy and Pierre Augier},
title = {{FluidFFT}: Common {API} (C$\mathplus\mathplus$ and Python)
    for Fast Fourier Transform {HPC} Libraries},
journal = {Journal of Open Research Software}
}
"""


__all__ = [
    "__citation__",
    "__version__",
    "byte_align",
    "create_fft_object",
    "empty_aligned",
    "get_module_fullname_from_method",
    "get_plugins",
    "get_methods",
    "import_fft_class",
]


_plugins = None


def get_plugins(reload=False, ndim=None, sequential=None):
    """Discover the fluidfft plugins installed"""
    global _plugins
    if _plugins is None or reload:
        _plugins = entry_points(group="fluidfft.plugins")

    if not _plugins:
        raise RuntimeError("No Fluidfft plugins were found.")

    if ndim is None and sequential is None:
        return _plugins

    if ndim is None:
        index = 6
        prefix = ""
    elif ndim in (2, 3):
        index = 0
        prefix = f"fft{ndim}d."
    else:
        raise ValueError(f"Unsupported value for {ndim = }")

    if sequential is not None and not sequential:
        prefix += "mpi_"
    elif sequential:
        prefix += "with_"

    return tuple(
        plugin for plugin in _plugins if plugin.name[index:].startswith(prefix)
    )


def get_methods(ndim=None, sequential=None):
    """Get available methods"""
    plugins = get_plugins(ndim=ndim, sequential=sequential)
    return set(plug.name for plug in plugins)


def get_module_fullname_from_method(method):
    """Get the module name from a method string

    Parameters
    ----------

    method : str
      Name of module or string characterizing a method.

    """
    plugins = get_plugins()
    selected_plugins = plugins.select(name=method)
    if len(selected_plugins) == 0:
        raise ValueError(
            f"Cannot find a fluidfft plugin for {method = }. {plugins}"
        )
    elif len(selected_plugins) > 1:
        logging.warning(
            f"{len(selected_plugins)} plugins were found for {method = }"
        )

    return selected_plugins[method].value


def _normalize_method_name(method):
    """Normalize a method name"""
    if method == "sequential":
        method = "fft2d.with_fftw2d"
    elif method.startswith("fluidfft:"):
        method = method.removeprefix("fluidfft:")
    return method


def _check_failure(method):
    """Check if a tiny fft maker can be created"""

    if not any(method.endswith(postfix) for postfix in ("pfft", "p3dfft")):
        return False

    if os.environ.get("FLUIDFFT_DISABLE_IMPORT_CHECK"):
        return False

    # for few methods, try before real import because importing can lead to
    # a fatal error (Illegal instruction)
    if mpi.rank == 0:
        if mpi.nb_proc > 1:
            # We need to filter out the MPI environment variables.
            # Fragile because it is specific to MPI implementations
            env = {
                key: value
                for key, value in os.environ.items()
                if not (
                    "MPI" in key
                    or key.startswith("PMI_")
                    or key.startswith("PMIX_")
                )
            }
        else:
            env = os.environ
        try:
            # TODO: capture stdout and stderr and include last line in case of failure
            subprocess.check_call(
                [
                    sys.executable,
                    "-c",
                    f"from fluidfft import create_fft_object as c; c('{method}', 2, 2, 2, check=False)",
                ],
                env=env,
                shell=False,
            )
            failure = False
        except subprocess.CalledProcessError:
            failure = True

    else:
        failure = None

    if mpi.nb_proc > 1:
        failure = mpi.comm.bcast(failure, root=0)

    return failure


def import_fft_class(method, raise_import_error=True, check=True):
    """Import a fft class.

    Parameters
    ----------

    method : str
      Name of module or string characterizing a method. It has to correspond to
      a module of fluidfft. The first part "fluidfft." of the module "path" can
      be omitted.

    raise_import_error : {True}, False

      If raise_import_error == False and if there is an import error, the
      function handles the error and returns None.

    Returns
    -------

    The corresponding FFT class.

    """

    if isinstance(method, EntryPoint):
        module_fullname = method.value
        method = method.name
    else:
        method = _normalize_method_name(method)
        module_fullname = get_module_fullname_from_method(method)

    if check:
        failure = _check_failure(method)
        if failure:
            if not raise_import_error:
                mpi.printby0("ImportError during check:", module_fullname)
                return None
            else:
                raise ImportError(module_fullname)

    try:
        mod = importlib.import_module(module_fullname)
    except ImportError:
        if raise_import_error:
            raise

        mpi.printby0("ImportError:", module_fullname)
        return None

    return mod.FFTclass


def _get_classes(ndim, sequential):
    plugins = get_plugins(ndim=ndim, sequential=sequential)
    return {
        plugin.name: import_fft_class(plugin, raise_import_error=False)
        for plugin in plugins
    }


def create_fft_object(method, n0, n1, n2=None, check=True):
    """Helper for creating fft objects.

    Parameters
    ----------

    method : str
      Name of module or string characterizing a method. It has to correspond to
      a module of fluidfft. The first part "fluidfft." of the module "path" can
      be omitted.

    n0, n1, n2 : int
      Dimensions of the real space array (in sequential).

    Returns
    -------

    The corresponding FFT object.


    """

    cls = import_fft_class(method, check=check)

    str_module = cls.__module__

    if n2 is None and "fft3d" in str_module:
        raise ValueError("Arguments incompatible")

    if n2 is not None and "fft2d" in str_module:
        n2 = None

    if n2 is None:
        return cls(n0, n1)

    else:
        return cls(n0, n1, n2)
