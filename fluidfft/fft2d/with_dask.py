
"""Class using Dask (:mod:`fluidfft.fft2d.with_dask`)
=====================================================

.. autoclass:: FFT2DWithDASK
   :members:

TODO: Find a mechanism for setting chunksize

"""
import warnings


import numpy as np
import dask.array as da
from pyfftw.interfaces import dask_fft, cache

from fluiddyn.calcul.easypyfft import FFTW2DReal2Complex


class FFT2DWithDASK(FFTW2DReal2Complex):
    def __init__(self, nx, ny):
        warnings.warn(
                "The `with_dask` FFT class is a prototype and not fully "
                "functional yet."
        )
        shapeX = (ny, nx)

        shapeK = list(shapeX)
        shapeK[-1] = shapeK[-1] // 2 + 1
        shapeK = tuple(shapeK)

        self.shapeX = shapeX
        self.shapeK = self.shapeK_seq = self.shapeK_loc = shapeK
        self.coef_norm = da.prod(np.array(shapeX))
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

    def fft(self, fieldX):
        if isinstance(fieldX, np.ndarray):
            fieldX = da.asarray(fieldX)

        fieldK = dask_fft.rfft2(fieldX)
        return fieldK / self.coef_norm

    def ifft(self, fieldK):
        if isinstance(fieldK, np.ndarray):
            fieldK = da.asarray(fieldK)

        fieldX = dask_fft.irfft2(fieldK)
        return fieldX

    def fft_as_arg(self, field, field_fft):
        field_fft[:] = self.fft(field)

    def ifft_as_arg(self, field_fft, field):
        field[:] = self.ifft(field_fft)

    def compute_energy_from_Fourier(self, ff_fft):
        result = (
            da.sum(abs(ff_fft[:, 0]) ** 2 + abs(ff_fft[:, -1]) ** 2)
            + 2 * da.sum(abs(ff_fft[:, 1:-1]) ** 2)
        ) / 2
        return result

    def compute_energy_from_spatial(self, ff):
        result = da.mean(abs(ff) ** 2) / 2
        return result

    def run_tests(self):
        arr = da.random.random_sample(self.shapeX)
        arr_fft = self.fft(arr)
        arr = self.ifft(arr_fft)
        arr_fft = self.fft(arr)

        nrj = self.compute_energy_from_spatial(arr)
        nrj_fft = self.compute_energy_from_Fourier(arr_fft)

        assert da.allclose(nrj, nrj_fft).compute()

        arr2_fft = np.zeros(self.shapeK, dtype=np.complex128)
        self.fft_as_arg(arr, arr2_fft)
        nrj2_fft = self.compute_energy_from_Fourier(arr2_fft)
        assert da.allclose(nrj, nrj2_fft).compute()

        arr2 = np.empty(self.shapeX)
        self.ifft_as_arg(arr_fft, arr2)
        nrj2 = self.compute_energy_from_spatial(arr2)
        assert da.allclose(nrj, nrj2).compute()

    compute_energy_from_K = compute_energy_from_Fourier


FFTclass = FFT2DWithDASK
