
#include <complex.h>
#include <fftw3-mpi.h>

#include <base_fft2dmpi.h>


class FFT2DMPIWithFFTWMPI2D: public BaseFFT2DMPI
{
 public:
  FFT2DMPIWithFFTWMPI2D(int N0, int N1);
  ~FFT2DMPIWithFFTWMPI2D();
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
  int nX1_pad;
  int coef_norm;
  fftw_plan plan_r2c, plan_c2r;
  double *arrayX;
  fftw_complex *arrayK;
  ptrdiff_t alloc_local, local_K0_start;
  ptrdiff_t local_X0_start;

  unsigned flags;
};
