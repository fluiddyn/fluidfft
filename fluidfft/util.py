"""Utilities for FluidFFT
=========================

"""
# import inspect as _inspect

import numpy as np

# we need this try because this file is executed during the build when we don't
# have fluiddyn
try:
    from fluidfft import byte_align
except ImportError:
    pass


def can_import(pkg_name, check_version=None):
    """Checks if a package can be imported."""
    from importlib import import_module

    try:
        pkg = import_module(pkg_name)
    except ImportError:
        return False

    else:
        if check_version is not None:
            if pkg.__version__ < check_version:
                raise ValueError(
                    "Please upgrade to {} >= {}".format(pkg_name, check_version)
                )

        return True


# pa: since this code is not tested and not yet used, I comment it to increase
# the coverage.

# # FIXME: When the next version of Pythran is released
# # use_pythran = can_import('pythran', '0.8.4')
# use_pythran = can_import('pythran')


# def from_cython(func=None, name=None, module=None):
#     """Ensures compatibility to use cython function as an argument.
#     When used with Pythran >= 0.8.4, returns this returns a PyCapsule.
#     On all other cases, this returns the function itself.

#     Similar to ``scipy.LowLevelCallable.from_cython`` method.

#     Parameters
#     ----------
#     func : function, optional
#         A cython function with C API exported. Optional, because ``cdef``
#         functions cannot be imported into Python. In that scenario, specify
#         name and module instead.

#     name : str, optional
#         Name of the cython function

#     module : module, optional
#         Module which contains the cython function

#     Returns
#     -------
#     PyCapsule or function

#     """
#     if use_pythran:
#         if name is None:
#             name = func.__name__.split('.')[-1]

#         if module is None:
#             module = _inspect.getmodule(func)

#         try:
#             return module.__pyx_capi__[name]
#         except AttributeError:
#             raise ValueError(
#                 ('{} is not a Cython module with __pyx_capi__'
#                  'attribute').format(module))
#         except KeyError:
#             raise ValueError(
#                 'No function {!r} found in __pyx_capi__ of the module'.format(
#                     name))
#     else:
#         return func


def _rescale_random(values, min_val=None, max_val=None):
    if min_val is None and max_val is None:
        return byte_align(values)

    if min_val is None:
        min_val = 0
    elif max_val is None:
        max_val = 1
    if min_val > max_val:
        raise ValueError("min_val > max_val")

    if np.iscomplexobj(min_val):
        raise ValueError("np.iscomplexobj(min_val)")

    if np.iscomplexobj(max_val):
        raise ValueError("np.iscomplexobj(max_val)")

    values = abs(max_val - min_val) * values + min_val
    if np.iscomplexobj(values):
        values += 1j * min_val
    return byte_align(values)
