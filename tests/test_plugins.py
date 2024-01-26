from importlib_metadata import entry_points

from fluidfft import get_plugins


methodss = {
    (2, True): set(
        [
            "fft2d.with_fftw1d",
            "fft2d.with_fftw2d",
            "fft2d.with_cufft",
            "fft2d.with_pyfftw",
            "fft2d.with_dask",
        ]
    ),
    (2, False): set(
        [
            "fft2d.mpi_with_fftw1d",
            "fft2d.mpi_with_fftwmpi2d",
        ]
    ),
    (3, True): set(
        [
            "fft3d.with_fftw3d",
            "fft3d.with_pyfftw",
            "fft3d.with_cufft",
        ]
    ),
    (3, False): set(
        [
            "fft3d.mpi_with_fftw1d",
            "fft3d.mpi_with_fftwmpi3d",
            "fft3d.mpi_with_p3dfft",
            "fft3d.mpi_with_pfft",
        ]
    ),
}


def _methods_from_plugins(plugins):
    return set(plug.name for plug in plugins)


def _get_methods(ndim=None, sequential=None):
    return _methods_from_plugins(get_plugins(ndim=ndim, sequential=sequential))


def test_plugins():
    plugins = get_plugins()
    assert plugins

    for ndim in (2, 3):
        assert _get_methods(ndim=ndim) == methodss[(ndim, True)].union(
            methodss[(ndim, False)]
        )
        for sequential in (True, False):
            assert methodss[(ndim, sequential)] == _get_methods(
                ndim=ndim, sequential=sequential
            )
