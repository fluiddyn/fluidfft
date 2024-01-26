"""Operators Base (:mod:`fluidfft.fft2d.base`)

=================================================

.. autoclass:: OperatorsBase
   :members:
   :undoc-members:

"""

import numpy as np
import fluidfft


class OperatorsBase:
    """An abstract base class for sharing methods across operator classes."""

    def empty_aligned(self, shape, dtype="float64", order="C", n=None):
        try:
            return self.opfft.empty_aligned(shape, dtype)
        except AttributeError:
            return fluidfft.empty_aligned(shape, dtype, order, n)

    def byte_align(self, array, n=None, dtype=None):
        try:
            return self.opfft.byte_align(array, dtype)
        except AttributeError:
            return fluidfft.byte_align(array, n, dtype)

    def _rescale_random(self, values, min_val=None, max_val=None):
        byte_align = self.byte_align

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
