from .mpi_with_mpi4pyfft import FFT3DMPIWithMPI4PYFFT


class FFT3DMPIWithMPI4PYFFTSlab(FFT3DMPIWithMPI4PYFFT):
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

    _slab = True


FFTclass = FFT3DMPIWithMPI4PYFFTSlab

if __name__ == "__main__":
    offt = FFTclass(6, 14, 4)
    offt.print_summary_for_debug()
    offt.run_tests()
    offt.run_benchs()
