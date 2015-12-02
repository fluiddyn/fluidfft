
#include <complex.h>
#include <fftw3.h>

#include <base_fft2d.h>

class FFT2DWithFFTW2D: public BaseFFT2D
{
 public:
  FFT2DWithFFTW2D(int N0, int N1);
  ~FFT2DWithFFTW2D();
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
  fftw_plan plan_r2c, plan_c2r;
  double *arrayX;
  fftw_complex *arrayK;

  unsigned flags;
};
