
#include <stdio.h>
#include <complex.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <base_fft2d.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#ifdef SINGLE_PREC
  typedef float2 dcomplex;
#else
  typedef double2 dcomplex;
#endif

class FFT2DWithCUFFT: public BaseFFT2D
{
 public:
  FFT2DWithCUFFT(int N0, int N1);
  ~FFT2DWithCUFFT();
  void destroy();
  
  void fft(myreal *fieldX, mycomplex *fieldK);
  void ifft(mycomplex *fieldK, myreal *fieldX);
  myreal compute_energy_from_K(mycomplex* fieldK);
  myreal compute_mean_from_K(mycomplex* fieldK);


  virtual const char* get_classname();
  
  myreal compute_energy_from_X(myreal* fieldX);
  myreal compute_mean_from_X(myreal* fieldX);

  void init_array_X_random(myreal* &fieldX);

 private:
  int nX1loc, nK1loc, nXxloc, nXyloc, nXx, nXy, nKyloc;
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
