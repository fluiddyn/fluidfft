
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

  void fft(myreal *fieldX, mycomplex *fieldK);
  void ifft(mycomplex *fieldK, myreal *fieldX);
  
  myreal compute_energy_from_X(myreal* fieldX);
  myreal compute_energy_from_K(mycomplex* fieldK);
  myreal compute_mean_from_X(myreal* fieldX);
  myreal compute_mean_from_K(mycomplex* fieldK);

  void init_array_X_random(myreal* &fieldX);

 private:
  int coef_norm;
  myfftw_plan plan_r2c, plan_c2r;
  myreal *arrayX;
  mycomplex *arrayK;

  unsigned flags;
};
