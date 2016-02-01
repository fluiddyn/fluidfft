
#include <iostream>

#include <cstdlib>

using namespace std;

#include <fft2dmpi_with_fftwmpi2d.h>
#include <fft2d_with_fftw2d.h>
#ifndef SINGLE_PREC
#include <fft2dmpi_with_fftw1d.h>
#endif
#include <fft2d_with_fftw1d.h>

#ifdef CUDA
#include <fft2d_with_cufft.h>
#endif
const int N0default=16, N1default=16;

void parse_args(int nb_args, char **argv, int &N0, int &N1)
{
  int i;
  std::string prefixN0("--N0="), prefixN1("--N1="), arg;
  
  N0 = N0default;
  N1 = N1default;

  for (i=0; i<nb_args; i++)
    {
      arg = argv[i];
      // cout << "argv[i]:" << arg << endl; 

      if (!arg.compare(0, prefixN0.size(), prefixN0))
	N0 = atoi(arg.substr(prefixN0.size()).c_str());

      if (!arg.compare(0, prefixN1.size(), prefixN1))
	N1 = atoi(arg.substr(prefixN1.size()).c_str());
    }
}


int main(int argc, char **argv)
{
  int N0, N1, nb_procs;

  parse_args(argc, argv, N0, N1);
  
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &(nb_procs));
  
#ifndef SINGLE_PREC
  FFT2DMPIWithFFTW1D s(N0, N1);
  s.test();
  s.bench();
  s.bench();
  s.destroy();
#endif
  FFT2DMPIWithFFTWMPI2D s2(N0, N1);
  s2.test();
  s2.bench();
  s2.bench();
  s2.destroy();

  if (nb_procs == 1)
    {
      FFT2DWithFFTW1D s1(N0, N1);
      s1.test();
      s1.bench();
      s1.bench();
      s1.destroy();

      FFT2DWithFFTW2D s3(N0, N1);
      s3.test();
      s3.bench();
      s3.bench();
      s3.destroy();

#ifdef CUDA
       FFT2DWithCUFFT s5(N0, N1);
       s5.test();
       s5.bench();
       s5.bench();
       s5.destroy();
#endif
    }
  
  MPI_Finalize();

  return 0;
}
