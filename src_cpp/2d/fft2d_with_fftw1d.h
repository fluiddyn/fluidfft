
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

  void fft(myreal *fieldX, mycomplex *fieldK);
  void ifft(mycomplex *fieldK, myreal *fieldX);
  myreal compute_energy_from_K(mycomplex* fieldK);
  myreal compute_mean_from_K(mycomplex* fieldK);
  myreal compute_energy_from_X(myreal* fieldX);
  myreal compute_mean_from_X(myreal* fieldX);

  void init_array_X_random(myreal* &fieldX);

 private:
  int coef_norm;
  myreal *arrayX;
  mycomplex *arrayK_pR, *arrayK_pC;
  myfftw_plan plan_r2c, plan_c2c_fwd, plan_c2r, plan_c2c_bwd;

  unsigned flags;
};
