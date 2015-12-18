
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
  
  virtual const char* get_classname();

  /* int get_local_size_X(); */
  /* int get_local_size_K(); */

  void fft(double *fieldX, fftw_complex *fieldK);
  void ifft(fftw_complex *fieldK, double *fieldX);
  
  double compute_energy_from_X(double* fieldX);
  double compute_energy_from_K(fftw_complex* fieldK);
  double compute_mean_from_X(double* fieldX);
  double compute_mean_from_K(fftw_complex* fieldK);

  void init_array_X_random(double* &fieldX);

 private:
//__global__ void  vectorNorm(const double norm, fftw_complex *A, int numElements)
  int nX2loc, nK2loc, nXxloc, nXyloc, nXzloc, nKzloc, nXx, nXy, nXz, nKyloc;
  int coef_norm;
  double *arrayX;
  double *arrayK;
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
  //fftw_complex *data;
  typedef double2 dcomplex;
  dcomplex *data;
  double *datar;
  cufftHandle plan;
  cufftHandle plan1;
};
