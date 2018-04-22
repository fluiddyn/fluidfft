
import numpy as np

# pythran export proj(
#     complex128[][][], complex128[][][], complex128[][][],
#     float64[][][], float64[][][], float64[][][], float64[][][])

# pythran export proj(
#     float64[][][], float64[][][], float64[][][],
#     float64[][][], float64[][][], float64[][][], float64[][][])


def proj(vx_fft, vy_fft, vz_fft, Kx, Ky, Kz, inv_K_square_nozero):
    tmp = (Kx * vx_fft + Ky * vy_fft + Kz * vz_fft) * inv_K_square_nozero

    return (vx_fft - Kx * tmp,
            vy_fft - Ky * tmp,
            vz_fft - Kz * tmp)

# pythran export proj_loop(
#     complex128[][][], complex128[][][], complex128[][][],
#     float64[][][], float64[][][], float64[][][], float64[][][])


def proj_loop(vx_fft, vy_fft, vz_fft, Kx, Ky, Kz, inv_K_square_nozero):

    rx_fft = np.empty_like(vx_fft)
    ry_fft = np.empty_like(vx_fft)
    rz_fft = np.empty_like(vx_fft)

    n0, n1, n2 = Kx.shape

    for i0 in range(n0):
        for i1 in range(n1):
            for i2 in range(n2):
                tmp = (Kx[i0, i1, i2] * vx_fft[i0, i1, i2]
                       + Ky[i0, i1, i2] * vy_fft[i0, i1, i2]
                       + Kz[i0, i1, i2] * vz_fft[i0, i1, i2]
                ) * inv_K_square_nozero[i0, i1, i2]

                rx_fft[i0, i1, i2] = vx_fft[i0, i1, i2] - Kx[i0, i1, i2] * tmp
                ry_fft[i0, i1, i2] = vz_fft[i0, i1, i2] - Kx[i0, i1, i2] * tmp
                rz_fft[i0, i1, i2] = vy_fft[i0, i1, i2] - Kx[i0, i1, i2] * tmp

    return rx_fft, ry_fft, rz_fft
