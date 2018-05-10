/* compute_energy.cpp */

#include <iostream>

using namespace std;

#include <fft3dmpi_with_fftw1d.h>
// #include <fft3dmpi_with_fftwmpi3d.h>
// #include <fft3dmpi_with_pfft.h>
// #include <fft3dmpi_with_p3dfft.h>


int main(int argc, char **argv)
{
  int N0, N1, N2;
  N0 = N1 = N2 = 16;
  // MPI-related
  int nb_procs = 2;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &(nb_procs));

  myreal* array_X;
  mycomplex* array_K;
  myreal energy_X, energy_K, energy_X_after_ifft;

  FFT3DMPIWithFFTW1D o(N0, N1, N2);
  // FFT3DMPIWithFFTWMPI3D o(N0, N1, N2);
  // FFT3DMPIWithPFFT o(N0, N1, N2);
  // FFT3DMPIWithP3DFFT o(N0, N1, N2);

  o.init_array_X_random(array_X);  // Initialize physical array with random values
  o.alloc_array_K(array_K);  // Allocate spectral array in memory
  o.fft(array_X, array_K);  // Forward FFT
  energy_X = o.compute_energy_from_X(array_X);
  energy_K = o.compute_energy_from_K(array_K);

  o.ifft(array_K, array_X);  // Inverse FFT
  energy_X_after_ifft = o.compute_energy_from_X(array_X);

  cout<<"Energies in physical array (before, after):"
      <<energy_X<<", "<<energy_X_after_ifft<<endl;
  cout<<"Energies in spectral array :"
      <<energy_K<<endl;
  MPI_Finalize();
  return 0;
}
