from fluidfft import get_plugins, get_methods


methodss = {
    (2, True): set(
        [
            "fft2d.with_pyfftw",
            "fft2d.with_dask",
        ]
    ),
    (2, False): set(),
    (3, True): set(
        [
            "fft3d.with_pyfftw",
        ]
    ),
    (3, False): set(),
}


try:
    import fluidfft_fftw
except ImportError:
    pass
else:
    for method in ("fft2d.with_fftw1d", "fft2d.with_fftw2d"):
        methodss[2, True].add(method)

    methodss[3, True].add("fft3d.with_fftw3d")
    del fluidfft_fftw


try:
    import fluidfft_mpi_with_fftw
except ImportError:
    pass
else:
    methodss[2, False].add("fft2d.mpi_with_fftw1d")
    methodss[3, False].add("fft3d.mpi_with_fftw1d")
    del fluidfft_mpi_with_fftw

try:
    import fluidfft_fftwmpi
except ImportError:
    pass
else:
    methodss[2, False].add("fft2d.mpi_with_fftwmpi2d")
    methodss[3, False].add("fft3d.mpi_with_fftwmpi3d")
    del fluidfft_fftwmpi


try:
    import fluidfft_pfft
except ImportError:
    pass
else:
    methodss[3, False].add("fft3d.mpi_with_pfft")
    del fluidfft_pfft

try:
    import fluidfft_p3dfft
except ImportError:
    pass
else:
    methodss[3, False].add("fft3d.mpi_with_p3dfft")
    del fluidfft_p3dfft


def test_plugins():
    plugins = get_plugins()
    assert plugins

    for ndim in (2, 3):
        assert sorted(get_methods(ndim=ndim)) == sorted(
            methodss[(ndim, True)].union(methodss[(ndim, False)])
        )
        for sequential in (True, False):
            assert methodss[(ndim, sequential)] == get_methods(
                ndim=ndim, sequential=sequential
            )
