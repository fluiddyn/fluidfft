
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

  void fft(real_cu *fieldX, myfftw_complex *fieldK);
  void ifft(myfftw_complex *fieldK, real_cu *fieldX);
  real_cu compute_energy_from_K(myfftw_complex* fieldK);
  real_cu compute_mean_from_K(myfftw_complex* fieldK);
  void sum_wavenumbers_complex(myfftw_complex* fieldK, myfftw_complex* result);

  real_cu compute_energy_from_X(real_cu* fieldX);
  real_cu compute_mean_from_X(real_cu* fieldX);

  real_cu sum_wavenumbers_double(real_cu* fieldK);
  
  void init_array_X_random(real_cu* &fieldX);

 private:
  int coef_norm;
  real_cu *arrayX;
  myfftw_plan plan_r2c, plan_c2r;
  myfftw_complex *arrayK;

  unsigned flags;
};
