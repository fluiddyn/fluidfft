
#include <stdio.h>
#include <complex.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <base_fft3d.h>
#include <helper_functions.h>
#include <helper_cuda.h>


class FFT3DWithCUFFT: public BaseFFT3D
{
 public:
  FFT3DWithCUFFT(int N0, int N1, int N2);
  ~FFT3DWithCUFFT();
  void destroy();
  
#ifdef SINGLE_PREC
  typedef float2 dcomplex;
  void fft(real_cu *fieldX, fftwf_complex *fieldK);
  void ifft(fftwf_complex *fieldK, real_cu *fieldX);
  real_cu compute_energy_from_K(fftwf_complex* fieldK);
  real_cu compute_mean_from_K(fftwf_complex* fieldK);
#else
  typedef double2 dcomplex;
  void fft(real_cu *fieldX, fftw_complex *fieldK);
  void ifft(fftw_complex *fieldK, real_cu *fieldX);
  real_cu compute_energy_from_K(fftw_complex* fieldK);
  real_cu compute_mean_from_K(fftw_complex* fieldK);
#endif

  virtual const char* get_classname();

  /* int get_local_size_X(); */
  /* int get_local_size_K(); */

  
  real_cu compute_energy_from_X(real_cu* fieldX);
  real_cu compute_mean_from_X(real_cu* fieldX);

  void init_array_X_random(real_cu* &fieldX);

 private:
//__global__ void  vectorNorm(const double norm, fftw_complex *A, int numElements)
  int nX2loc, nK2loc, nXxloc, nXyloc, nXzloc, nKzloc, nXx, nXy, nXz, nKyloc;
  int coef_norm;
  ptrdiff_t alloc_local;

  ptrdiff_t N[3];
  ptrdiff_t local_ni[3], local_i_start[3];
  ptrdiff_t local_no[3], local_o_start[3];

  ptrdiff_t local_K0_start, local_K1_start;
  ptrdiff_t local_X0_start, local_X1_start;
  
  unsigned flags;
  int mem_size;//equivalent à la taille de arrayK?
  int mem_sizer;//equivalent à la taille de arrayK?

// Allocate device memory for signal
  real_cu *arrayX;
  real_cu *arrayK;
  dcomplex *data;
  real_cu *datar;
  cufftHandle plan;
  cufftHandle plan1;
};
