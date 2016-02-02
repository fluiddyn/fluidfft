
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
  
#ifdef SINGLE_PREC
  void fft(myreal *fieldX, fftwf_complex *fieldK);
  void ifft(fftwf_complex *fieldK, myreal *fieldX);
  myreal compute_energy_from_K(fftwf_complex* fieldK);
  myreal compute_mean_from_K(fftwf_complex* fieldK);
#else
  void fft(myreal *fieldX, fftw_complex *fieldK);
  void ifft(fftw_complex *fieldK, myreal *fieldX);
  myreal compute_energy_from_K(fftw_complex* fieldK);
  myreal compute_mean_from_K(fftw_complex* fieldK);
#endif

  virtual const char* get_classname();
  
  myreal compute_energy_from_X(myreal* fieldX);
  myreal compute_mean_from_X(myreal* fieldX);

  void init_array_X_random(myreal* &fieldX);

 private:
  int nX2loc, nK2loc, nXxloc, nXyloc, nXzloc, nKzloc, nXx, nXy, nXz, nKyloc;
  int coef_norm;

  int mem_size;//equivalent à la taille de arrayK?
  int mem_sizer;//equivalent à la taille de arrayK?

// Allocate device memory for signal
  myreal *arrayX;
  myreal *arrayK;
  dcomplex *data;
  myreal *datar;
  cufftHandle plan;
  cufftHandle plan1;
};
