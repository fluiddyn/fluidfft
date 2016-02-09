
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

  void fft(myreal *fieldX, mycomplex *fieldK);
  void ifft(mycomplex *fieldK, myreal *fieldX);

  myreal compute_energy_from_K(mycomplex* fieldK);
  myreal compute_mean_from_K(mycomplex* fieldK);
  void sum_wavenumbers_complex(mycomplex* fieldK, mycomplex* result);

  myreal compute_energy_from_X(myreal* fieldX);
  myreal compute_mean_from_X(myreal* fieldX);
  myreal sum_wavenumbers_double(myreal* fieldK);

  void init_array_X_random(myreal* &fieldX);

  virtual void get_dimX_K(int*, int*, int*);
  virtual void get_seq_indices_first_K(int*, int*);

 private:
  int nX2loc, nK2loc, nXxloc, nXyloc, nXzloc, nKzloc, nXx, nXy, nXz;
  int coef_norm;
#ifdef SINGLE_PREC
  pfftf_plan plan_r2c, plan_c2r;
#else
  pfft_plan plan_r2c, plan_c2r;
#endif  
  mycomplex *arrayK;
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
