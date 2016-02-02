
#include <stdio.h>
#include <complex.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <base_fft3d.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#ifdef SINGLE_PREC
  typedef float2 dcomplex;
#else
  typedef double2 dcomplex;
#endif

class FFT3DWithCUFFT: public BaseFFT3D
{
 public:
  FFT3DWithCUFFT(int N0, int N1, int N2);
  ~FFT3DWithCUFFT();
  void destroy();

  void fft(real_cu *fieldX, myfftw_complex *fieldK);
  void ifft(myfftw_complex *fieldK, real_cu *fieldX);
  real_cu compute_energy_from_K(myfftw_complex* fieldK);
  real_cu compute_mean_from_K(myfftw_complex* fieldK);

  virtual const char* get_classname();
  
  real_cu compute_energy_from_X(real_cu* fieldX);
  real_cu compute_mean_from_X(real_cu* fieldX);

  void init_array_X_random(real_cu* &fieldX);

 private:
  int nX2loc, nK2loc, nXxloc, nXyloc, nXzloc, nKzloc, nXx, nXy, nXz, nKyloc;
  int coef_norm;

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
