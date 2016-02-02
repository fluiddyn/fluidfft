
#include <complex.h>
#include <pfft.h>

#include <base_fft3dmpi.h>


class FFT3DMPIWithPFFT: public BaseFFT3DMPI
{
 public:
  FFT3DMPIWithPFFT(int N0, int N1, int N2);
  ~FFT3DMPIWithPFFT();
  void destroy();
  
  virtual const char* get_classname();

#ifdef SINGLE_PREC
  void fft(myreal *fieldX, fftwf_complex *fieldK);
  void ifft(fftwf_complex *fieldK, myreal *fieldX);
  myreal compute_energy_from_K(fftwf_complex* fieldK);
  myreal compute_mean_from_K(fftwf_complex* fieldK);
  void sum_wavenumbers_complex(fftwf_complex* fieldK, fftwf_complex* result);
#else
  void fft(myreal *fieldX, fftw_complex *fieldK);
  void ifft(fftw_complex *fieldK, myreal *fieldX);
  myreal compute_energy_from_K(fftw_complex* fieldK);
  myreal compute_mean_from_K(fftw_complex* fieldK);
  void sum_wavenumbers_complex(fftw_complex* fieldK, fftw_complex* result);
#endif

  myreal compute_energy_from_X(myreal* fieldX);
  myreal compute_mean_from_X(myreal* fieldX);
  myreal sum_wavenumbers_double(myreal* fieldK);

  void init_array_X_random(myreal* &fieldX);

  virtual void get_dimX_K(int*, int*, int*);
  virtual void get_seq_index_first_K(int*, int*);

 private:
  int nX2loc, nK2loc, nXxloc, nXyloc, nXzloc, nKzloc, nXx, nXy, nXz;
  int coef_norm;
#ifdef SINGLE_PREC
  pfftf_plan plan_r2c, plan_c2r;
  fftwf_complex *arrayK;
#else
  pfft_plan plan_r2c, plan_c2r;
  fftw_complex *arrayK;
#endif
  myreal *arrayX;
  ptrdiff_t alloc_local;

  int nprocmesh[2];
  ptrdiff_t N[3];
  ptrdiff_t local_ni[3], local_i_start[3];
  ptrdiff_t local_no[3], local_o_start[3];

  ptrdiff_t local_K0_start, local_K1_start;
  ptrdiff_t local_X0_start, local_X1_start;
  
  MPI_Comm comm_cart_2d;
  
  unsigned flags;
};
