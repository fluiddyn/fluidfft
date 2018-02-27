
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
