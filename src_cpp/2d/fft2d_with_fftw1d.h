
#include <complex.h>
#include <fftw3.h>

#include <base_fft2d.h>

class FFT2DWithFFTW1D: public BaseFFT2D
{
 public:
  FFT2DWithFFTW1D(int N0, int N1);
  ~FFT2DWithFFTW1D();
  void destroy();

  virtual const char* get_classname();

  void fft(real_cu *fieldX, myfftw_complex *fieldK);
  void ifft(myfftw_complex *fieldK, real_cu *fieldX);
  real_cu compute_energy_from_K(myfftw_complex* fieldK);
  real_cu compute_mean_from_K(myfftw_complex* fieldK);
  real_cu compute_energy_from_X(real_cu* fieldX);
  real_cu compute_mean_from_X(real_cu* fieldX);

  void init_array_X_random(real_cu* &fieldX);

 private:
  int coef_norm;
  real_cu *arrayX;
  myfftw_complex *arrayK_pR, *arrayK_pC;
  myfftw_plan plan_r2c, plan_c2c_fwd, plan_c2r, plan_c2c_bwd;

  unsigned flags;
};
