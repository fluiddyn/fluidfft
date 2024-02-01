
#include <base_fft2dmpi.h>

class FFT2DMPIWithFFTW1D : public BaseFFT2DMPI {
public:
  FFT2DMPIWithFFTW1D(int N0, int N1);
  ~FFT2DMPIWithFFTW1D();
  void destroy();

  virtual bool are_parameters_bad();

  virtual const char *get_classname();

  void fft(myreal *fieldX, mycomplex *fieldK);
  void ifft(mycomplex *fieldK, myreal *fieldX);

  myreal compute_energy_from_X(myreal *fieldX);
  myreal compute_energy_from_K(mycomplex *fieldK);
  myreal sum_wavenumbers(myreal *fieldK);
  myreal compute_mean_from_X(myreal *fieldX);
  myreal compute_mean_from_K(mycomplex *fieldK);

private:
  fftw_plan plan_r2c, plan_c2c_fwd, plan_c2r, plan_c2c_bwd;
  myreal *arrayX;
  mycomplex *arrayK_pR, *arrayK_pC;

  unsigned flags;
  MPI_Datatype MPI_type_column, MPI_type_block;
};
