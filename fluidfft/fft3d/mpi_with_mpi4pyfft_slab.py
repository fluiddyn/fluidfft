from .mpi_with_mpi4pyfft import FFT3dMPIWithMPI4PyFFT


class FFT3dMPIWithMPI4PyFFTSlab(FFT3dMPIWithMPI4PyFFT):
    _slab = False

    """Perform Fast Fourier Transform in 3D.

    Parameters
    ----------

    n0 : int

      Global size over the first dimension in spatial space. This corresponds
      to the z direction.

    n1 : int

      Global size over the second dimension in spatial space. This corresponds
      to the y direction.

    n2 : int

      Global size over the second dimension in spatial space. This corresponds
      to the x direction.

    """


FFTclass = FFT3dMPIWithMPI4PyFFTSlab
