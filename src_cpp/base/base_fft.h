
#ifndef _CLASS_BaseFFT
#define _CLASS_BaseFFT

#include <string.h>

#include <complex.h>
#include <fftw3.h>

#include <complex>
using std::complex;

#ifdef SINGLE_PREC
  typedef float real_cu;
#else
  typedef double real_cu;
#endif

class BaseFFT
{
 public:

  virtual void _init();
  virtual void _init_parallel();
  
  virtual const char* get_classname();
  
  virtual int test();
  virtual const char* bench(int nb_time_execute = 10);
 
#ifdef SINGLE_PREC
//  typedef float real_cu;
  virtual void fft(real_cu *fieldX, fftwf_complex *fieldK);
  virtual void ifft(fftwf_complex *fieldK, real_cu *fieldX);
  virtual real_cu compute_energy_from_K(fftwf_complex* fieldK);
  virtual real_cu compute_mean_from_K(fftwf_complex* fieldK);
  void alloc_array_K(fftwf_complex* &fieldK);  
#else
//  typedef double real_cu;
  virtual void fft(real_cu *fieldX, fftw_complex *fieldK);
  virtual void ifft(fftw_complex *fieldK, real_cu *fieldX);
  virtual real_cu compute_energy_from_K(fftw_complex* fieldK);
  virtual real_cu compute_mean_from_K(fftw_complex* fieldK);
  void alloc_array_K(fftw_complex* &fieldK);  
#endif

  
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
