

# pythran export project_perpk3d(
#     complex128[][][], complex128[][][], complex128[][][],
#     float64[][][], float64[][][], float64[][][], float64[][][])

def project_perpk3d(vx_fft, vy_fft, vz_fft, Kx, Ky, Kz, K_square_nozero):
    tmp = (Kx * vx_fft + Ky * vy_fft + Kz * vz_fft) / K_square_nozero

    return (vx_fft - Kx * tmp,
            vy_fft - Ky * tmp,
            vz_fft - Kz * tmp)
