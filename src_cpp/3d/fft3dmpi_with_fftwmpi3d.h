
#include <complex.h>
#include <fftw3-mpi.h>

#include <base_fft3dmpi.h>


class FFT3DMPIWithFFTWMPI3D: public BaseFFT3DMPI
{
 public:
  FFT3DMPIWithFFTWMPI3D(int N0, int N1, int N2);
  ~FFT3DMPIWithFFTWMPI3D();
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

  double sum_wavenumbers_double(double* fieldK);
  void sum_wavenumbers_complex(fftw_complex* fieldK, fftw_complex* result);
  
  void init_array_X_random(double* &fieldX);

  virtual void get_dimX_K(int*, int*, int*);
  virtual void get_seq_index_first_K(int*, int*);
  
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
