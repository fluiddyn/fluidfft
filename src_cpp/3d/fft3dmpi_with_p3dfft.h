#include <p3dfft.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <fftw3-mpi.h>

#include <base_fft3dmpi.h>


class FFT3DMPIWithP3DFFT: public BaseFFT3DMPI
{
 public:
  FFT3DMPIWithP3DFFT(int N0, int N1, int N2);
  ~FFT3DMPIWithP3DFFT();
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
  int nX2loc, nK2loc, nXxloc, nXyloc, nXzloc, nKzloc, nXx, nXy, nXz;
  int coef_norm;
  double *arrayX;
  double *arrayK;
  ptrdiff_t alloc_local;

  int nprocmesh[2];
  ptrdiff_t N[3];
  ptrdiff_t local_ni[3], local_i_start[3];
  ptrdiff_t local_no[3], local_o_start[3];

  ptrdiff_t local_K0_start, local_K1_start;
  ptrdiff_t local_X0_start, local_X1_start;
  
//  unsigned char op_f[3]="fft", op_b[3]="tff";
  
  unsigned flags;
};
