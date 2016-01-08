
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

  real_cu sum_wavenumbers_double(real_cu* fieldK);
  
  void init_array_X_random(real_cu* &fieldX);

 private:
  int coef_norm;
  real_cu *arrayX;
#ifdef SINGLE_PREC
  fftwf_plan plan_r2c, plan_c2r;
  fftwf_complex *arrayK;
#else
  fftw_plan plan_r2c, plan_c2r;
  fftw_complex *arrayK;
#endif

  unsigned flags;
};
