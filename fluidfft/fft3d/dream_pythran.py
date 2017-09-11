
# pythran export _vgradv_from_v2(
#     float64[][][], float64[][][], float64[][][],
#     complex128[][][], complex128[][][], complex128[][][],
#     float64[][][], float64[][][], float64[][][],
#     function(complex128[][][]) -> float64[][][])


def _vgradv_from_v2(vx, vy, vz, vx_fft, vy_fft, vz_fft,
                    Kx, Ky, Kz, ifft3d):
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
