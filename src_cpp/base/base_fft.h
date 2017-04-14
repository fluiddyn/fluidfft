
#ifndef _CLASS_BaseFFT
#define _CLASS_BaseFFT

#include <string.h>

#include <complex.h>
#include <fftw3.h>

//#include <math.h>
#include <complex>
using std::complex;

#ifdef SINGLE_PREC
  typedef float myreal;
  typedef fftwf_complex mycomplex;
  typedef fftwf_plan myfftw_plan;
#else
  typedef double myreal;
  typedef fftw_complex mycomplex;
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
 
  virtual void fft(myreal *fieldX, mycomplex *fieldK);
  virtual void ifft(mycomplex *fieldK, myreal *fieldX);
  virtual myreal compute_energy_from_K(mycomplex* fieldK);
  virtual myreal compute_mean_from_K(mycomplex* fieldK);
  void alloc_array_K(mycomplex* &fieldK);  

  
  virtual myreal compute_energy_from_X(myreal* fieldX);
  virtual myreal compute_mean_from_X(myreal* fieldX);

  virtual int get_local_size_X();
  virtual int get_local_size_K();
  
  virtual void init_array_X_random(myreal* &fieldX);

  virtual void alloc_array_X(myreal* &fieldX);

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
