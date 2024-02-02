from fluidfft import get_plugins, get_methods


methodss = {
    (2, True): set(
        [
            "fft2d.with_fftw1d",
            "fft2d.with_fftw2d",
            "fft2d.with_pyfftw",
            "fft2d.with_dask",
        ]
    ),
    (2, False): set(
        [
            "fft2d.mpi_with_fftw1d",
            # "fft2d.mpi_with_fftwmpi2d",
        ]
    ),
    (3, True): set(
        [
            "fft3d.with_fftw3d",
            "fft3d.with_pyfftw",
        ]
    ),
    (3, False): set(
        [
            "fft3d.mpi_with_fftw1d",
            # "fft3d.mpi_with_fftwmpi3d",
            # "fft3d.mpi_with_p3dfft",
            # "fft3d.mpi_with_pfft",
        ]
    ),
}


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
