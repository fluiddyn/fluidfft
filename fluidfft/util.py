"""Utilities for FluidFFT
=========================

"""
import inspect as _inspect


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
                raise ValueError('Please upgrade to {} >= {}'.format(
                    pkg_name, check_version))
        return True


# FIXME: When the next version of Pythran is released
# use_pythran = can_import('pythran', '0.8.4')
use_pythran = can_import('pythran')
use_capi = False


def from_cython(func=None, name=None, parent=None):
    """Ensures compatibility to use cython function as an argument.
    When used with Pythran >= 0.8.4, returns this returns a PyCapsule.
    On all other cases, this returns the function itself.

    Similar to ``scipy.LowLevelCallable.from_cython`` method.

    Parameters
    ----------
    func : function, optional
        A cython function with C API exported. Optional, because ``cdef``
        functions cannot be imported into Python. In that scenario, specify
        name and parent instead.

    name : str, optional
        Name of the cython function

    parent : parent, optional
        Module or instance which contains the cython function

    Returns
    -------
    PyCapsule or function

    """
    if use_pythran and use_capi:
        if name is None:
            name = func.__name__.split('.')[-1]

        if parent is None:
            parent = _inspect.getmodule(func)

        try:
            return parent.__pyx_capi__[name]
        except AttributeError:
            raise ValueError(
                ('{} is not a Cython module or instance with __pyx_capi__'
                 'attribute').format(parent))
        except KeyError:
            raise ValueError(
                'No function {!r} found in __pyx_capi__ of the parent'.format(
                    name))
    else:
        return func
