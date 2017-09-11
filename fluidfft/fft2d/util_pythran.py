"""

[pythran]
complex_hook = True

"""


# pythran export dealiasing_variable(complex128[][], uint8[][], int, int)

def dealiasing_variable(ff_fft, where, nK0loc, nK1loc):
    for iK0 in range(nK0loc):
        for iK1 in range(nK1loc):
            if where[iK0, iK1]:
                ff_fft[iK0, iK1] = 0.


# # pythran export dealiasing_variable2(complex128[][], uint8[][])

# def dealiasing_variable2(ff_fft, indexes_dealiased):
#         for i0, i1 in indexes_dealiased:
#             ff_fft[i0, i1] = 0.


# pythran export vecfft_from_rotfft(complex128[][], float64[][], float64[][])

def vecfft_from_rotfft(rot_fft, KX_over_K2, KY_over_K2):
    """Return the velocity in spectral space computed from the
    rotational."""
    ux_fft = 1j * KY_over_K2 * rot_fft
    uy_fft = -1j * KX_over_K2 * rot_fft
    return ux_fft, uy_fft


# pythran export gradfft_from_fft(complex128[][], float64[][], float64[][])

def gradfft_from_fft(f_fft, KX, KY):
    """Return the gradient of f_fft in spectral space."""
    px_f_fft = 1j * KX * f_fft
    py_f_fft = 1j * KY * f_fft
    return px_f_fft, py_f_fft


# pythran export divfft_from_vecfft(
#     complex128[][], complex128[][], float64[][], float64[][])

def divfft_from_vecfft(vecx_fft, vecy_fft, KX, KY):
    return 1j*(KX*vecx_fft + KY*vecy_fft)


# pythran export rotfft_from_vecfft(
#     complex128[][], complex128[][], float64[][], float64[][])


def rotfft_from_vecfft(vecx_fft, vecy_fft, KX, KY):
    return 1j*(KX*vecy_fft - KY*vecx_fft)