
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

  void fft(real_cu *fieldX, myfftw_complex *fieldK);
  void ifft(myfftw_complex *fieldK, real_cu *fieldX);
  
  real_cu compute_energy_from_X(real_cu* fieldX);
  real_cu compute_energy_from_K(myfftw_complex* fieldK);
  real_cu compute_mean_from_X(real_cu* fieldX);
  real_cu compute_mean_from_K(myfftw_complex* fieldK);

  void init_array_X_random(real_cu* &fieldX);

 private:
  int coef_norm;
  myfftw_plan plan_r2c, plan_c2r;
  real_cu *arrayX;
  myfftw_complex *arrayK;

  unsigned flags;
};
