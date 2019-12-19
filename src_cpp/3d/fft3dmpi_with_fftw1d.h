
#include <base_fft3dmpi.h>

class FFT3DMPIWithFFTW1D : public BaseFFT3DMPI {
public:
  FFT3DMPIWithFFTW1D(int N0, int N1, int N2);
  void destroy();

  bool are_parameters_bad();

  virtual const char *get_classname();

  void fft(myreal *fieldX, mycomplex *fieldK);
  void ifft(mycomplex *fieldK, myreal *fieldX);
  void ifft_destroy(mycomplex *fieldK, myreal *fieldX);

  myreal compute_energy_from_K(mycomplex *fieldK);

  myreal sum_wavenumbers_double(myreal *fieldK);
  void sum_wavenumbers_complex(mycomplex *fieldK, mycomplex *result);

  void get_dimX_K(int *, int *, int *);
  virtual void get_seq_indices_first_X(int *, int *, int *);
  virtual void get_seq_indices_first_K(int *, int *, int *);

  char is_transposed;
  ptrdiff_t local_X0_start, local_K0_start;

private:
#ifdef SINGLE_PREC
  fftwf_plan plan_r2c, plan_c2c_fwd, plan_c2r, plan_c2c_bwd;
  fftwf_plan plan_c2c1_fwd, plan_c2c1_bwd;
#else
  fftw_plan plan_r2c, plan_c2c_fwd, plan_c2r, plan_c2c_bwd;
  fftw_plan plan_c2c1_fwd, plan_c2c1_bwd;
#endif
  myreal *arrayX;
  mycomplex *arrayK_pR, *arrayK_pC;

  unsigned flags;
  MPI_Datatype MPI_type_column, MPI_type_block, MPI_type_block1,
      MPI_type_block2;
};
