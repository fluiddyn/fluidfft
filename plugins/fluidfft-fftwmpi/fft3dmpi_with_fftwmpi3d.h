
#include <fftw3-mpi.h>

#include <base_fft3dmpi.h>

class FFT3DMPIWithFFTWMPI3D : public BaseFFT3DMPI {
public:
  FFT3DMPIWithFFTWMPI3D(int N0, int N1, int N2);
  void destroy();

  bool are_parameters_bad();
  virtual const char *get_classname();

  void fft(myreal *fieldX, mycomplex *fieldK);
  void ifft(mycomplex *fieldK, myreal *fieldX);
  void ifft_destroy(mycomplex *fieldK, myreal *fieldX);

  myreal compute_energy_from_K(mycomplex *fieldK);
  myreal sum_wavenumbers_double(myreal *fieldK);
  void sum_wavenumbers_complex(mycomplex *fieldK, mycomplex *result);

  virtual void get_dimX_K(int *, int *, int *);
  virtual void get_seq_indices_first_X(int *, int *, int *);
  virtual void get_seq_indices_first_K(int *, int *, int *);

private:
  int nX1_pad;
  myreal *arrayX;
  mycomplex *arrayK;
  myfftw_plan plan_r2c, plan_c2r;
  ptrdiff_t alloc_local, local_K0_start, size_fieldK;
  ptrdiff_t local_X0_start;

  unsigned flags;
};
