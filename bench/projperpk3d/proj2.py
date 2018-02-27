
# pythran export proj(
#     float64[][][], float64[][][], float64[][][],
#     float64[][][], float64[][][], float64[][][], float64[][][])

# pythran export proj(
#     complex128[][][], complex128[][][], complex128[][][],
#     float64[][][], float64[][][], float64[][][], float64[][][])


def proj(vx_fft, vy_fft, vz_fft, Kx, Ky, Kz, inv_K_square_nozero):
    tmp = (Kx * vx_fft + Ky * vy_fft + Kz * vz_fft) * inv_K_square_nozero

    return (vx_fft - Kx * tmp,
            vy_fft - Ky * tmp,
            vz_fft - Kz * tmp)


# pythran export divfft_from_vecfft(
#     complex128[][][], complex128[][][], complex128[][][],
#     float64[][][], float64[][][], float64[][][])

def divfft_from_vecfft(vx_fft, vy_fft, vz_fft, kx, ky, kz):
    """Compute the divergence of a vector (in spectral space)"""
    return 1j * (kx * vx_fft + ky * vy_fft + kz * vz_fft)


# pythran export rotfft_from_vecfft(
#     complex128[][][], complex128[][][], complex128[][][],
#     float64[][][], float64[][][], float64[][][])

def rotfft_from_vecfft(vx_fft, vy_fft, vz_fft, kx, ky, kz):
    """Compute the curl of a vector (in spectral space)"""
    return (1j * (ky * vz_fft + kz * vy_fft),
            1j * (kz * vx_fft + kx * vz_fft),
            1j * (kx * vy_fft + ky * vx_fft))


# pythran export vector_product(
#     float64[][][], float64[][][], float64[][][],
#     float64[][][], float64[][][], float64[][][])

def vector_product(ax, ay, az, bx, by, bz):
    return (ay * bz - az * by,
            az * bx - ax * bz,
            ax * by - ay * bx)
