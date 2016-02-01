
#ifndef _CLASS_BaseFFT
#define _CLASS_BaseFFT

#include <string.h>

#include <complex.h>
#include <fftw3.h>

#include <complex>
using std::complex;

#ifdef SINGLE_PREC
  typedef float real_cu;
  typedef fftwf_complex myfftw_complex;
  typedef fftwf_plan myfftw_plan;
#else
  typedef double real_cu;
  typedef fftw_complex myfftw_complex;
  typedef fftw_plan myfftw_plan;
#endif

class BaseFFT
{
 public:

  virtual void _init();
  virtual void _init_parallel();
  
  virtual const char* get_classname();
  
  virtual int test();
  virtual const char* bench(int nb_time_execute = 10);
 
  virtual void fft(real_cu *fieldX, myfftw_complex *fieldK);
  virtual void ifft(myfftw_complex *fieldK, real_cu *fieldX);
  virtual real_cu compute_energy_from_K(myfftw_complex* fieldK);
  virtual real_cu compute_mean_from_K(myfftw_complex* fieldK);
  void alloc_array_K(myfftw_complex* &fieldK);  

  
  virtual real_cu compute_energy_from_X(real_cu* fieldX);
  virtual real_cu compute_mean_from_X(real_cu* fieldX);

  virtual int get_local_size_X();
  virtual int get_local_size_K();
  
  virtual void init_array_X_random(real_cu* &fieldX);

  virtual void alloc_array_X(real_cu* &fieldX);

  int rank, nb_proc;

protected:
  /* X and K denote physical and Fourier spaces. */
  /* in physical space: */
  /* y corresponds to dim 0 */
  /* x corresponds to dim 1 */
  int N0, N1, nX0, nX1, nX0loc;
  int ny, nx, nXyloc;
  /* in Fourier space */
  /* y corresponds to dim 1 */
  /* x corresponds to dim 0 */
  int nK0, nK1, nK0loc; 
  int nKx, nKy, nKxloc;

};

#endif
