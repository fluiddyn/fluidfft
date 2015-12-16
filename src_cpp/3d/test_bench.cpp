
#include <iostream>
#include <cstdlib>

using namespace std;

#include <mpi.h>

#include <fft3d_with_fftw3d.h>
#include <fft3dmpi_with_fftwmpi3d.h>
#include <fft3dmpi_with_pfft.h>
#include <fft3dmpi_with_p3dfft.h>

const int N0default=16, N1default=16, N2default=16;

void parse_args(int nb_args, char **argv, int &N0, int &N1, int &N2)
{
  int i;
  std::string prefixN0("--N0="), prefixN1("--N1="), prefixN2("--N2=");
  std::string prefixN("--N="), arg;
  
  N0 = N0default;
  N1 = N1default;
  N2 = N2default;

  for (i=0; i<nb_args; i++)
    {
      arg = argv[i];
      // cout << "argv[i]:" << arg << endl; 

      if (!arg.compare(0, prefixN0.size(), prefixN0))
	N0 = atoi(arg.substr(prefixN0.size()).c_str());

      if (!arg.compare(0, prefixN1.size(), prefixN1))
	N1 = atoi(arg.substr(prefixN1.size()).c_str());

      if (!arg.compare(0, prefixN2.size(), prefixN2))
	N2 = atoi(arg.substr(prefixN2.size()).c_str());

      if (!arg.compare(0, prefixN.size(), prefixN))
	{
	  N0 = atoi(arg.substr(prefixN.size()).c_str());
	  N1 = N0;
	  N2 = N0;
	}
    }
}


int main(int argc, char **argv)
{
  int N0, N1, N2, nb_procs;

  parse_args(argc, argv, N0, N1, N2);
  
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &(nb_procs));

  FFT3DMPIWithFFTWMPI3D s(N0, N1, N2);
  s.test();
  s.bench();
  s.bench();
  s.destroy();

  FFT3DMPIWithPFFT s2(N0, N1, N2);
  s2.test();
  s2.bench();
  s2.bench();
  s2.destroy();
  
  FFT3DMPIWithP3DFFT s3(N0, N1, N2);
  s3.test();
  s3.bench();
  s3.bench();
  s3.destroy();
  // if (nb_procs == 1)
  //   {
  //     FFT3DWithFFTW3D s3(N0, N1, N2);
  //     s3.test();
  //     s3.bench();
  //     s3.bench();
  //     s3.destroy();
  //   }
  
  MPI_Finalize();

  return 0;
}
