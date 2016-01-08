
#include <complex.h>
#include <fftw3-mpi.h>

#include <base_fft3dmpi.h>


class FFT3DMPIWithFFTWMPI3D: public BaseFFT3DMPI
{
 public:
  FFT3DMPIWithFFTWMPI3D(int N0, int N1, int N2);
  ~FFT3DMPIWithFFTWMPI3D();
  void destroy();
  
  virtual const char* get_classname();

  /* int get_local_size_X(); */
  /* int get_local_size_K(); */

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

  void init_array_X_random(real_cu* &fieldX);

  real_cu sum_wavenumbers_double(real_cu* fieldK);
  
  virtual void get_dimX_K(int*, int*, int*);
  virtual void get_seq_index_first_K(int*, int*);
  
 private:
  int nX1_pad;
  int coef_norm;
  real_cu *arrayX;
#ifdef SINGLE_PREC
  fftwf_complex *arrayK;
  fftwf_plan plan_r2c, plan_c2r;
#else
  fftw_complex *arrayK;
  fftw_plan plan_r2c, plan_c2r;
#endif
  ptrdiff_t alloc_local, local_K0_start;
  ptrdiff_t local_X0_start;

  unsigned flags;
};
