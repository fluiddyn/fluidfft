
#include <complex.h>
#include <fftw3.h>

#include <mpi.h>

#include <base_fft2dmpi.h>

class FFT2DMPIWithFFTW1D: public BaseFFT2DMPI
{
 public:
  FFT2DMPIWithFFTW1D(int N0, int N1);
  ~FFT2DMPIWithFFTW1D();
  void destroy();
  
  virtual const char* get_classname();
  
  void fft(double *fieldX, fftw_complex *fieldK);
  void ifft(fftw_complex *fieldK, double *fieldX);
  
  double compute_energy_from_X(double* fieldX);
  double compute_energy_from_K(fftw_complex* fieldK);
  double compute_mean_from_X(double* fieldX);
  double compute_mean_from_K(fftw_complex* fieldK);

  void init_array_X_random(double* &fieldX);

 private:
  int coef_norm;
  fftw_plan plan_r2c, plan_c2c_fwd, plan_c2r, plan_c2c_bwd;
  double *arrayX;
  fftw_complex *arrayK_pR, *arrayK_pC;

  unsigned flags;
  MPI_Datatype MPI_type_column, MPI_type_block; 
};
