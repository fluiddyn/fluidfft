
#include <base_fft3d.h>

class FFT3DWithFFTW3D : public BaseFFT3D {
public:
  FFT3DWithFFTW3D(int N0, int N1, int N2);
  ~FFT3DWithFFTW3D();
  void destroy();

  virtual const char *get_classname();

  void fft(myreal *fieldX, mycomplex *fieldK);
  void ifft(mycomplex *fieldK, myreal *fieldX);
  void ifft_destroy(mycomplex *fieldK, myreal *fieldX);

  myreal compute_energy_from_K(mycomplex *fieldK);
  myreal compute_mean_from_K(mycomplex *fieldK);
  void sum_wavenumbers_complex(mycomplex *fieldK, mycomplex *result);

  myreal compute_energy_from_X(myreal *fieldX);
  myreal sum_wavenumbers_double(myreal *fieldK);

private:
  myreal *arrayX;

  myfftw_plan plan_r2c, plan_c2r;
  mycomplex *arrayK;

  unsigned flags;
};
