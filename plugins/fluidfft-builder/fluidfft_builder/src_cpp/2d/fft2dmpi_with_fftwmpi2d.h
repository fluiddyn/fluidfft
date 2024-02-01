
#include <fftw3-mpi.h>

#include <base_fft2dmpi.h>

class FFT2DMPIWithFFTWMPI2D : public BaseFFT2DMPI {
public:
  FFT2DMPIWithFFTWMPI2D(int N0, int N1);
  ~FFT2DMPIWithFFTWMPI2D();
  void destroy();

  virtual bool are_parameters_bad();

  virtual const char *get_classname();

  /* int get_local_size_X(); */
  /* int get_local_size_K(); */

  void fft(myreal *fieldX, mycomplex *fieldK);
  void ifft(mycomplex *fieldK, myreal *fieldX);

  myreal compute_energy_from_X(myreal *fieldX);
  myreal compute_energy_from_K(mycomplex *fieldK);
  myreal sum_wavenumbers(myreal *fieldK);
  myreal compute_mean_from_X(myreal *fieldX);
  myreal compute_mean_from_K(mycomplex *fieldK);

private:
  int nX1_pad, last_rank_nozero;
  myfftw_plan plan_r2c, plan_c2r;
  myreal *arrayX;
  mycomplex *arrayK;
  ptrdiff_t alloc_local, size_fieldK;

  unsigned flags;
};
