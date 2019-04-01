"""Class using Dask (:mod:`fluidfft.fft2d.with_dask`)
=====================================================

.. autoclass:: FFT2DWithDASK
   :members:

TODO: Find a mechanism for setting chunksize

"""
import warnings
from typing import Union

import numpy as np
import dask.array as da
from pyfftw.interfaces import dask_fft, cache

from fluiddyn.calcul.easypyfft import FFTW2DReal2Complex


DaskOrNumpyArray = Union[da.core.Array, np.ndarray]


class FFT2DWithDASK(FFTW2DReal2Complex):
    """Perform Fast Fourier Transform in 2d using ``dask.array.fft`` interface of
    pyFFTW.

    Parameters
    ----------

    n0 : int

      Global size over the first dimension in spatial space. This corresponds
      to the y direction.

    n1 : int

      Global size over the second dimension in spatial space. This corresponds
      to the x direction.

    """

    def __init__(self, n0=2, n1=2):
        warnings.warn(
            "The `with_dask` FFT class is a prototype and not fully "
            "functional yet."
        )
        shapeX = (n0, n1)

        shapeK = list(shapeX)
        shapeK[-1] = shapeK[-1] // 2 + 1
        shapeK = tuple(shapeK)

        self.shapeX = shapeX
        self.shapeK = self.shapeK_seq = self.shapeK_loc = shapeK
        self.coef_norm = np.prod(shapeX)
        self.inv_coef_norm = 1.0 / self.coef_norm

        self.chunks = "auto"

        # The cache temporarily stores a copy of any interim pyfftw.FFTW
        # objects that are created
        cache.enable()
        # Set keepalive time in seconds. Default: 0.1
        cache.set_keepalive_time(60)

        self.fft2d = self.fft  # dask_fft.rfft2
        self.ifft2d = self.ifft  # dask_fft.irfft2

        self.ifft_as_arg_destroy = self.ifft_as_arg
        self.empty_aligned = da.empty

    def fft(self, fieldX: DaskOrNumpyArray) -> da.core.Array:
        if isinstance(fieldX, np.ndarray):
            fieldX = da.asarray(fieldX)

        fieldK = dask_fft.rfft2(fieldX)
        return fieldK * self.inv_coef_norm

    def ifft(self, fieldK: DaskOrNumpyArray) -> da.core.Array:
        if isinstance(fieldK, np.ndarray):
            fieldK = da.asarray(fieldK)

        fieldX = dask_fft.irfft2(fieldK)
        return fieldX * self.coef_norm

    def fft_as_arg(self, field: DaskOrNumpyArray, field_fft: np.ndarray) -> None:
        field_fft[:] = self.fft(field)

    def ifft_as_arg(self, field_fft: DaskOrNumpyArray, field: np.ndarray) -> None:
        field[:] = self.ifft(field_fft)

    def compute_energy_from_Fourier(self, ff_fft: DaskOrNumpyArray) -> float:
        result = (
            da.sum(abs(ff_fft[:, 0]) ** 2 + abs(ff_fft[:, -1]) ** 2)
            + 2 * da.sum(abs(ff_fft[:, 1:-1]) ** 2)
        ) / 2
        return result.compute()

    def compute_energy_from_spatial(self, ff: DaskOrNumpyArray) -> float:
        result = da.mean(abs(ff) ** 2) / 2
        return result.compute()

    compute_energy_from_K = compute_energy_from_Fourier
    compute_energy_from_X = compute_energy_from_spatial


FFTclass = FFT2DWithDASK
