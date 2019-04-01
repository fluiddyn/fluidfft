"""Operators Base (:mod:`fluidfft.fft2d.base`)

=================================================

.. autoclass:: OperatorsBase
   :members:
   :undoc-members:

"""
import numpy as np
import pyfftw


class OperatorsBase:
    """An abstract base class for sharing methods across operator classes."""

    def __init__(self):
        try:
            self.empty_aligned = self.opfft.empty_aligned
        except AttributeError:
            self.empty_aligned = pyfftw.empty_aligned

        try:
            self.byte_align = self.opfft.byte_align
        except AttributeError:
            self.byte_align = pyfftw.byte_align


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
