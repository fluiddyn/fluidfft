
#include <complex.h>
#include <fftw3.h>

#include <base_fft3d.h>

class FFT3DWithFFTW3D: public BaseFFT3D
{
 public:
  FFT3DWithFFTW3D(int N0, int N1, int N2);
  ~FFT3DWithFFTW3D();
  void destroy();
  
  virtual const char* get_classname();

  void fft(double *fieldX, fftw_complex *fieldK);
  void ifft(fftw_complex *fieldK, double *fieldX);
  
  double compute_energy_from_X(double* fieldX);
  double compute_energy_from_K(fftw_complex* fieldK);
  double compute_mean_from_X(double* fieldX);
  double compute_mean_from_K(fftw_complex* fieldK);

  double sum_wavenumbers_double(double* fieldK);
  void sum_wavenumbers_complex(fftw_complex* fieldK, fftw_complex* result);
  
  void init_array_X_random(double* &fieldX);

 private:
  int coef_norm;
  fftw_plan plan_r2c, plan_c2r;
  double *arrayX;
  fftw_complex *arrayK;

  unsigned flags;
};
