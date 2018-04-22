
# pythran export proj(
#     complex128[][][], complex128[][][], complex128[][][],
#     float64[][][], float64[][][], float64[][][], float64[][][])


def proj(vx_fft, vy_fft, vz_fft, Kx, Ky, Kz, inv_K_square_nozero):
    tmp = (Kx * vx_fft + Ky * vy_fft + Kz * vz_fft) * inv_K_square_nozero

    vx_fft -= Kx * tmp
    vy_fft -= Ky * tmp
    vz_fft -= Kz * tmp


# pythran export proj_loop(
#     complex128[][][], complex128[][][], complex128[][][],
#     float64[][][], float64[][][], float64[][][], float64[][][])


def proj_loop(vx_fft, vy_fft, vz_fft, Kx, Ky, Kz, inv_K_square_nozero):

    n0, n1, n2 = Kx.shape

    for i0 in range(n0):
        for i1 in range(n1):
                for i2 in range(n2):
                    tmp = (Kx[i0, i1, i2] * vx_fft[i0, i1, i2]
                           + Ky[i0, i1, i2] * vy_fft[i0, i1, i2]
                           + Kz[i0, i1, i2] * vz_fft[i0, i1, i2]
                    ) * inv_K_square_nozero[i0, i1, i2]

                    vx_fft[i0, i1, i2] -= Kx[i0, i1, i2] * tmp
                    vy_fft[i0, i1, i2] -= Ky[i0, i1, i2] * tmp
                    vz_fft[i0, i1, i2] -= Kz[i0, i1, i2] * tmp
