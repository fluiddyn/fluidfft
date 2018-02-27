"""

[pythran]
complex_hook = True

"""


# pythran export gradfft_from_fft(complex128[][], float64[][], float64[][])

def gradfft_from_fft(f_fft, KX, KY):
    """Return the gradient of f_fft in spectral space."""
    px_f_fft = 1j * KX * f_fft
    py_f_fft = 1j * KY * f_fft
    return px_f_fft, py_f_fft
