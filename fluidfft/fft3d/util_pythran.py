
import numpy as np


# pythran export project_perpk3d(
#     float64[][][], float64[][][], float64[][][],
#     float64[][][], float64[][][], float64[][][], float64[][][])

# pythran export project_perpk3d(
#     complex128[][][], complex128[][][], complex128[][][],
#     float64[][][], float64[][][], float64[][][], float64[][][])


def project_perpk3d(vx_fft, vy_fft, vz_fft, Kx, Ky, Kz, inv_K_square_nozero):
    # tmp = (Kx * vx_fft + Ky * vy_fft + Kz * vz_fft) * inv_K_square_nozero

    # vx_fft -= Kx * tmp
    # vy_fft -= Ky * tmp
    # vz_fft -= Kz * tmp

    n0, n1, n2 = vx_fft.shape

    for i0 in range(n0):
        for i1 in range(n1):
            for i2 in range(n2):
                kx = Kx[i0, i1, i2]
                ky = Ky[i0, i1, i2]
                kz = Kz[i0, i1, i2]

                tmp = (
                    kx
                    * vx_fft[i0, i1, i2]
                    + ky
                    * vy_fft[i0, i1, i2]
                    + kz
                    * vz_fft[i0, i1, i2]
                ) * inv_K_square_nozero[
                    i0, i1, i2
                ]

                vx_fft[i0, i1, i2] -= kx * tmp
                vy_fft[i0, i1, i2] -= ky * tmp
                vz_fft[i0, i1, i2] -= kz * tmp


# pythran export divfft_from_vecfft(
#     complex128[][][], complex128[][][], complex128[][][],
#     float64[][][], float64[][][], float64[][][])


def divfft_from_vecfft(vx_fft, vy_fft, vz_fft, kx, ky, kz):
    """Compute the divergence of a vector (in spectral space)"""
    return 1j * (kx * vx_fft + ky * vy_fft + kz * vz_fft)


# pythran export rotfft_from_vecfft(
#     complex128[][][], complex128[][][], complex128[][][],
#     float64[][][], float64[][][], float64[][][])


def rotfft_from_vecfft(vx_fft, vy_fft, vz_fft, Kx, Ky, Kz):
    """Compute the curl of a vector (in spectral space)"""
    # return (1j * (Ky * vz_fft - Kz * vy_fft),
    #         1j * (Kz * vx_fft - Kx * vz_fft),
    #         1j * (Kx * vy_fft - Ky * vx_fft))

    rotxfft = np.empty_like(vx_fft)
    rotyfft = np.empty_like(vx_fft)
    rotzfft = np.empty_like(vx_fft)

    n0, n1, n2 = vx_fft.shape

    for i0 in range(n0):
        for i1 in range(n1):
            for i2 in range(n2):
                kx = Kx[i0, i1, i2]
                ky = Ky[i0, i1, i2]
                kz = Kz[i0, i1, i2]
                vx = vx_fft[i0, i1, i2]
                vy = vy_fft[i0, i1, i2]
                vz = vz_fft[i0, i1, i2]

                rotxfft[i0, i1, i2] = 1j * (ky * vz - kz * vy)
                rotyfft[i0, i1, i2] = 1j * (kz * vx - kx * vz)
                rotzfft[i0, i1, i2] = 1j * (kx * vy - ky * vx)

    return rotxfft, rotyfft, rotzfft


# pythran export rotfft_from_vecfft_outin(
#     complex128[][][], complex128[][][], complex128[][][],
#     float64[][][], float64[][][], float64[][][],
#     complex128[][][], complex128[][][], complex128[][][])


def rotfft_from_vecfft_outin(
    vx_fft, vy_fft, vz_fft, Kx, Ky, Kz, rotxfft, rotyfft, rotzfft
):
    """Compute the curl of a vector (in spectral space)"""
    # return (1j * (Ky * vz_fft - Kz * vy_fft),
    #         1j * (Kz * vx_fft - Kx * vz_fft),
    #         1j * (Kx * vy_fft - Ky * vx_fft))

    n0, n1, n2 = vx_fft.shape

    for i0 in range(n0):
        for i1 in range(n1):
            for i2 in range(n2):
                kx = Kx[i0, i1, i2]
                ky = Ky[i0, i1, i2]
                kz = Kz[i0, i1, i2]
                vx = vx_fft[i0, i1, i2]
                vy = vy_fft[i0, i1, i2]
                vz = vz_fft[i0, i1, i2]

                rotxfft[i0, i1, i2] = 1j * (ky * vz - kz * vy)
                rotyfft[i0, i1, i2] = 1j * (kz * vx - kx * vz)
                rotzfft[i0, i1, i2] = 1j * (kx * vy - ky * vx)


# pythran export vector_product(
#     float64[][][], float64[][][], float64[][][],
#     float64[][][], float64[][][], float64[][][])


def vector_product(ax, ay, az, bx, by, bz):
    """Compute the vector product.

    Warning: the arrays bx, by, bz are overwritten.

    """
    n0, n1, n2 = ax.shape

    for i0 in range(n0):
        for i1 in range(n1):
            for i2 in range(n2):
                elem_ax = ax[i0, i1, i2]
                elem_ay = ay[i0, i1, i2]
                elem_az = az[i0, i1, i2]
                elem_bx = bx[i0, i1, i2]
                elem_by = by[i0, i1, i2]
                elem_bz = bz[i0, i1, i2]

                bx[i0, i1, i2] = elem_ay * elem_bz - elem_az * elem_by
                by[i0, i1, i2] = elem_az * elem_bx - elem_ax * elem_bz
                bz[i0, i1, i2] = elem_ax * elem_by - elem_ay * elem_bx

    return bx, by, bz


# pythran export loop_spectra3d(float64[][][], float64[], float64[][][])


def loop_spectra3d(spectrum_k0k1k2, ks, K2):
    """Compute the 3d spectrum."""
    deltak = ks[1]
    nk = len(ks)
    spectrum3d = np.zeros(nk)
    nk0, nk1, nk2 = spectrum_k0k1k2.shape
    for ik0 in range(nk0):
        for ik1 in range(nk1):
            for ik2 in range(nk2):
                value = spectrum_k0k1k2[ik0, ik1, ik2]
                kappa = np.sqrt(K2[ik0, ik1, ik2])
                ik = int(kappa / deltak)
                if ik >= nk - 1:
                    ik = nk - 1
                    spectrum3d[ik] += value
                else:
                    coef_share = (kappa - ks[ik]) / deltak
                    spectrum3d[ik] += (1 - coef_share) * value
                    spectrum3d[ik + 1] += coef_share * value

    return spectrum3d
