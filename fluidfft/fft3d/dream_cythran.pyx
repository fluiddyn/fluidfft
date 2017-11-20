# cython: np_pythran=True
"""Attempt to use Pythran as a Numpy backend for Cython. If it works, an
advantage will be to create compiled extensions without explicitly writing
loops.

Without the macro (i.e. using Cython alone), this extension works.
Ideally it should work just by adding the macro to use Pythran.

It seems to build, but it creates a possibly bloated extension with the current
syntax, and also takes too long to build.

Another option will be to get rid of the class and define the function protocol
of ifft3d (see alternate implementation). However passing a function is not
allowed, just like in Pythran.

"""
cimport numpy as np


ctypedef np.float64_t DTYPEf_t
ctypedef np.complex128_t DTYPEc_t


cdef class OperatorsPseudoSpectral3D(object):
    cdef public object ifft3d

    def vgradv_from_v2(
            self,
            np.ndarray[DTYPEf_t, ndim=3] vx, np.ndarray[DTYPEf_t, ndim=3] vy,
            np.ndarray[DTYPEf_t, ndim=3] vz, np.ndarray[DTYPEc_t, ndim=3] vx_fft,
            np.ndarray[DTYPEc_t, ndim=3] vy_fft, np.ndarray[DTYPEc_t, ndim=3] vz_fft,
            np.ndarray[DTYPEf_t, ndim=3] Kx, np.ndarray[DTYPEf_t, ndim=3] Ky,
            np.ndarray[DTYPEf_t, ndim=3] Kz):

        px_vx_fft = 1j * Kx * vx_fft
        py_vx_fft = 1j * Ky * vx_fft
        pz_vx_fft = 1j * Kz * vx_fft

        px_vy_fft = 1j * Kx * vy_fft
        py_vy_fft = 1j * Ky * vy_fft
        pz_vy_fft = 1j * Kz * vy_fft

        px_vz_fft = 1j * Kx * vz_fft
        py_vz_fft = 1j * Ky * vz_fft
        pz_vz_fft = 1j * Kz * vz_fft

        vgradvx = (vx * self.ifft3d(px_vx_fft) +
                   vy * self.ifft3d(py_vx_fft) +
                   vz * self.ifft3d(pz_vx_fft))

        vgradvy = (vx * self.ifft3d(px_vy_fft) +
                   vy * self.ifft3d(py_vy_fft) +
                   vz * self.ifft3d(pz_vy_fft))

        vgradvz = (vx * self.ifft3d(px_vz_fft) +
                   vy * self.ifft3d(py_vz_fft) +
                   vz * self.ifft3d(pz_vz_fft))

        return vgradvx, vgradvy, vgradvz


"""
ctypedef np.ndarray[DTYPEf_t, ndim=3] (*ifft3d_t)(np.ndarray[DTYPEc_t, ndim=3])


def vgradv_from_v2(
        np.ndarray[DTYPEf_t, ndim=3] vx, np.ndarray[DTYPEf_t, ndim=3] vy,
        np.ndarray[DTYPEf_t, ndim=3] vz, np.ndarray[DTYPEc_t, ndim=3] vx_fft,
        np.ndarray[DTYPEc_t, ndim=3] vy_fft, np.ndarray[DTYPEc_t, ndim=3] vz_fft,
        np.ndarray[DTYPEf_t, ndim=3] Kx, np.ndarray[DTYPEf_t, ndim=3] Ky,
        np.ndarray[DTYPEf_t, ndim=3] Kz, ifft3d_t ifft3d):

    px_vx_fft = 1j * Kx * vx_fft
    py_vx_fft = 1j * Ky * vx_fft
    pz_vx_fft = 1j * Kz * vx_fft

    px_vy_fft = 1j * Kx * vy_fft
    py_vy_fft = 1j * Ky * vy_fft
    pz_vy_fft = 1j * Kz * vy_fft

    px_vz_fft = 1j * Kx * vz_fft
    py_vz_fft = 1j * Ky * vz_fft
    pz_vz_fft = 1j * Kz * vz_fft

    vgradvx = (vx * ifft3d(px_vx_fft) +
               vy * ifft3d(py_vx_fft) +
               vz * ifft3d(pz_vx_fft))

    vgradvy = (vx * ifft3d(px_vy_fft) +
               vy * ifft3d(py_vy_fft) +
               vz * ifft3d(pz_vy_fft))

    vgradvz = (vx * ifft3d(px_vz_fft) +
               vy * ifft3d(py_vz_fft) +
               vz * ifft3d(pz_vz_fft))

    return vgradvx, vgradvy, vgradvz
"""
