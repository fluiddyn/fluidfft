#include <p3dfft.h>
#include <base_fft3dmpi.h>

class FFT3DMPIWithP3DFFT: public BaseFFT3DMPI
{
 public:
  FFT3DMPIWithP3DFFT(int N0, int N1, int N2);
  ~FFT3DMPIWithP3DFFT();
  void destroy();
  
  virtual const char* get_classname();

  /* int get_local_size_X(); */
  /* int get_local_size_K(); */


  void fft(myreal *fieldX, mycomplex *fieldK);
  void ifft(mycomplex *fieldK, myreal *fieldX);

  myreal compute_energy_from_K(mycomplex* fieldK);
  myreal compute_mean_from_K(mycomplex* fieldK);
  void sum_wavenumbers_complex(mycomplex* fieldK, mycomplex* result);

  myreal sum_wavenumbers_double(myreal* fieldK);
  
 private:
  myreal coef_norm;
  myreal *arrayX;
  myreal *arrayK;
  ptrdiff_t alloc_local;

  int nprocmesh[2];
  ptrdiff_t N[3];
  ptrdiff_t local_ni[3], local_i_start[3];
  ptrdiff_t local_no[3], local_o_start[3];

  ptrdiff_t local_K0_start, local_K1_start;
  ptrdiff_t local_X0_start, local_X1_start;
  
//  unsigned char op_f[3]="fft", op_b[3]="tff";
  
  unsigned flags;
};
