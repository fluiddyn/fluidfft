
#include <cstdlib>
#include <iostream>

using namespace std;

#include <mpi.h>

#include <fft3d_with_fftw3d.h>
#include <fft3dmpi_with_fftw1d.h>
#include <fft3dmpi_with_fftwmpi3d.h>

#ifdef PFFT
#include <fft3dmpi_with_pfft.h>
#endif

#ifdef P3DFFT
#include <fft3dmpi_with_p3dfft.h>
#endif

#ifdef CUDA
#include <fft3d_with_cufft.h>
#endif
const int N0default = 16, N1default = 16, N2default = 16;

void parse_args(int nb_args, char **argv, int &N0, int &N1, int &N2) {
  int i;
  std::string prefixN0("--N0="), prefixN1("--N1="), prefixN2("--N2=");
  std::string prefixN("--N="), arg;

  N0 = N0default;
  N1 = N1default;
  N2 = N2default;

  for (i = 0; i < nb_args; i++) {
    arg = argv[i];
    // cout << "argv[i]:" << arg << endl;

    if (!arg.compare(0, prefixN0.size(), prefixN0))
      N0 = atoi(arg.substr(prefixN0.size()).c_str());

    if (!arg.compare(0, prefixN1.size(), prefixN1))
      N1 = atoi(arg.substr(prefixN1.size()).c_str());

    if (!arg.compare(0, prefixN2.size(), prefixN2))
      N2 = atoi(arg.substr(prefixN2.size()).c_str());

    if (!arg.compare(0, prefixN.size(), prefixN)) {
      N0 = atoi(arg.substr(prefixN.size()).c_str());
      N1 = N0;
      N2 = N0;
    }
  }
}

int main(int argc, char **argv) {
  int N0, N1, N2, nb_procs;
  myreal *times = new myreal[2];
  int nt = 10;

  parse_args(argc, argv, N0, N1, N2);

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &(nb_procs));

  FFT3DMPIWithFFTWMPI3D s(N0, N1, N2);
  s.test();
  s.bench(nt, times);
  s.bench(nt, times);
  s.destroy();

  FFT3DMPIWithFFTW1D s1(N0, N1, N2);
  s1.test();
  s1.bench(nt, times);
  s1.bench(nt, times);
  s1.destroy();

  if (nb_procs == 1) {
    FFT3DWithFFTW3D s4(N0, N1, N2);
    s4.test();
    s4.bench(nt, times);
    s4.bench(nt, times);
    s4.destroy();

#ifdef P3DFFT
    FFT3DMPIWithP3DFFT s2(N0, N1, N2);
    s2.test();
    s2.bench(nt, times);
    s2.bench(nt, times);
    s2.destroy();
#endif
#ifdef PFFT
    FFT3DMPIWithPFFT s3(N0, N1, N2);
    s3.test();
    s3.bench(nt, times);
    s3.bench(nt, times);
    s3.destroy();
#endif
  }

  MPI_Finalize();

  return 0;
}
