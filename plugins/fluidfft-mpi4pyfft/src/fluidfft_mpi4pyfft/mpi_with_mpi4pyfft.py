import numpy as np

from mpi4py import MPI
from mpi4py_fft import PFFT, newDistArray

from fluidfft.fft3d.base import BaseFFTMPI


class FFT3DMPIWithMPI4PYFFT(BaseFFTMPI):
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

    _slab = False
    _axes = (0, 1, 2)

    def __init__(self, n0=2, n1=2, n2=4):
        self._mpifft = PFFT(
            MPI.COMM_WORLD,
            (n0, n1, n2),
            axes=(0, 1, 2),
            dtype=np.float,
            slab=self._slab,
            # warning: segfault with the mpi4py_fft backend
            # see bench/debug_mpi4py-fft/README.md
            backend="pyfftw",
        )

        self.shapeX = self.get_shapeX_loc()
        self.shapeK = self.get_shapeK_loc()

        super().__init__(n0, n1, n2)

    def create_arrayX(self, value=None, shape=None):
        """Return a constant array in real space."""
        return newDistArray(self._mpifft, False, val=value)

    def create_arrayK(self, value=None, shape=None):
        """Return a constant array in real space."""
        return newDistArray(self._mpifft, True, val=value)

    def fft_as_arg(self, fieldX, fieldK):
        """Perform FFT and put result in second argument."""
        self._mpifft.forward(input_array=fieldX, output_array=fieldK)

    def ifft_as_arg(self, fieldK, fieldX):
        """Perform iFFT and put result in second argument."""
        self._mpifft.backward(input_array=fieldK, output_array=fieldX)

    ifft_as_arg_destroy = ifft_as_arg

    def fft(self, fieldX):
        """Perform FFT and return the result."""
        return self._mpifft.forward(input_array=fieldX)

    def ifft(self, fieldK):
        """Perform iFFT and return the result."""
        return self._mpifft.backward(input_array=fieldK)

    def get_shapeX_loc(self):
        """Get the shape of the array in real space for this mpi process."""
        return self._mpifft.shape(False)

    def get_shapeK_loc(self):
        """Get the shape of the array in Fourier space for this mpi process."""
        return self._mpifft.shape(True)

    def get_shapeX_seq(self):
        """Get the shape of an array in real space for a sequential run."""
        return self._mpifft.global_shape(False)

    def get_shapeK_seq(self):
        """Get the shape of an array in Fourier space for a sequential run."""
        return self._mpifft.global_shape(True)

    def get_dimX_K(self):
        """Get the indices of the real space dimension in Fourier space."""
        # "mpi4py-fft never transposes axes. Not to transpose is one of the
        # main selling points for the algorithm, transposing is complicated!""
        # Mikael Mortensen
        return 0, 1, 2

    def get_seq_indices_first_K(self):
        """Get the "sequential" indices of the first number in Fourier space."""
        return tuple(slice_.start for slice_ in self._mpifft.local_slice(True))

    def get_seq_indices_first_X(self):
        """Get the "sequential" indices of the first number in real space."""
        return tuple(slice_.start for slice_ in self._mpifft.local_slice(False))

    # def gather_Xspace(self, ff_loc, root=0):
    #     """Gather an array in real space for a parallel run.
    #     """
    #     raise NotImplementedError

    # def scatter_Xspace(self, ff_seq, root=0):
    #     """Scatter an array in real space for a parallel run.

    #     """
    #     raise NotImplementedError

    # def sum_wavenumbers(self, fieldK):
    #     """Compute the sum over all wavenumbers."""
    #     raise NotImplementedError

    # def compute_energy_from_spatial(self, fieldX):
    #     """Compute the mean energy from a real space array."""
    #     raise NotImplementedError

    # def compute_energy_from_Fourier(self, fieldK):
    #     """Compute the mean energy from a Fourier space array."""
    #     raise NotImplementedError

    # compute_energy_from_X = compute_energy_from_spatial
    # compute_energy_from_K = compute_energy_from_Fourier


FFTclass = FFT3DMPIWithMPI4PYFFT


if __name__ == "__main__":
    offt = FFTclass(6, 14, 4)
    offt.print_summary_for_debug()
    offt.run_tests()
    offt.run_benchs()
