
#include <stdio.h>
#include <complex.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <base_fft3d.h>
/* #include <helper_functions.h> */
/* #include <helper_cuda.h> */

#define checkCudaErrors(x) x

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

  virtual const char* get_classname();
  
  void fft(myreal *fieldX, mycomplex *fieldK);
  void ifft(mycomplex *fieldK, myreal *fieldX);
  
  myreal compute_energy_from_K(mycomplex* fieldK);
  myreal compute_mean_from_K(mycomplex* fieldK);
  void sum_wavenumbers_complex(mycomplex* fieldK, mycomplex* result);

  myreal compute_energy_from_X(myreal* fieldX);
  myreal compute_mean_from_X(myreal* fieldX);
  myreal sum_wavenumbers_double(myreal* fieldK);

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
