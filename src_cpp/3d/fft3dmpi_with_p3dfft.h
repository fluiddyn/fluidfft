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


#ifdef SINGLE_PREC
  void fft(real_cu *fieldX, fftwf_complex *fieldK);
  void ifft(fftwf_complex *fieldK, real_cu *fieldX);
  real_cu compute_energy_from_K(fftwf_complex* fieldK);
  real_cu compute_mean_from_K(fftwf_complex* fieldK);
  void sum_wavenumbers_complex(fftwf_complex* fieldK, fftwf_complex* result);
#else
  void fft(real_cu *fieldX, fftw_complex *fieldK);
  void ifft(fftw_complex *fieldK, real_cu *fieldX);
  real_cu compute_energy_from_K(fftw_complex* fieldK);
  real_cu compute_mean_from_K(fftw_complex* fieldK);
  void sum_wavenumbers_complex(fftw_complex* fieldK, fftw_complex* result);
#endif

  real_cu compute_energy_from_X(real_cu* fieldX);
  real_cu compute_mean_from_X(real_cu* fieldX);

  void init_array_X_random(real_cu* &fieldX);

 private:
  int nX2loc, nK2loc, nXxloc, nXyloc, nXzloc, nKzloc, nXx, nXy, nXz;
  int coef_norm;
  real_cu *arrayX;
  real_cu *arrayK;
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
