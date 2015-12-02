
#ifndef _CLASS_BaseFFT
#define _CLASS_BaseFFT

#include <string.h>

#include <complex.h>
#include <fftw3.h>

#include <complex>
using std::complex;

class BaseFFT
{
 public:

  virtual void _init();
  virtual void _init_parallel();
  
  virtual const char* get_classname();
  
  virtual int test();
  virtual const char* bench(int nb_time_execute = 10);
  
  virtual void fft(double *fieldX, fftw_complex *fieldK);
  virtual void ifft(fftw_complex *fieldK, double *fieldX);
  
  virtual double compute_energy_from_X(double* fieldX);
  virtual double compute_energy_from_K(fftw_complex* fieldK);
  virtual double compute_mean_from_X(double* fieldX);
  virtual double compute_mean_from_K(fftw_complex* fieldK);

  virtual int get_local_size_X();
  virtual int get_local_size_K();
  
  virtual void init_array_X_random(double* &fieldX);

  virtual void alloc_array_X(double* &fieldX);
  void alloc_array_K(fftw_complex* &fieldK);  

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
