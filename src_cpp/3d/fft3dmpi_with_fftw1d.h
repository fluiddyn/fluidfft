
#include <complex.h>
#include <fftw3.h>

#include <mpi.h>

#include <base_fft3dmpi.h>

class FFT3DMPIWithFFTW1D: public BaseFFT3DMPI
{
 public:
  FFT3DMPIWithFFTW1D(int N0, int N1, int N2);
  ~FFT3DMPIWithFFTW1D();
  void destroy();
  
  virtual const char* get_classname();
  
  void fft(myreal *fieldX, mycomplex *fieldK);
  void ifft(mycomplex *fieldK, myreal *fieldX);
  
  myreal compute_energy_from_X(myreal* fieldX);
  myreal compute_energy_from_K(mycomplex* fieldK);
  myreal compute_mean_from_X(myreal* fieldX);
  myreal compute_mean_from_K(mycomplex* fieldK);

  myreal sum_wavenumbers_double(myreal* fieldK);
  void sum_wavenumbers_complex(mycomplex* fieldK, mycomplex* result);

  void init_array_X_random(myreal* &fieldX);

  int get_local_size_K();
  int get_local_size_X();

  char is_transposed;
  ptrdiff_t local_X0_start, local_K0_start;

 private:
  int coef_norm;
  fftw_plan plan_r2c, plan_c2c_fwd, plan_c2r, plan_c2c_bwd;
  fftw_plan plan_c2c1_fwd, plan_c2c1_bwd;
  myreal *arrayX;
  mycomplex *arrayK_pR, *arrayK_pC;

  unsigned flags;
  MPI_Datatype MPI_type_column, MPI_type_block, MPI_type_block1, MPI_type_block2; 


};
