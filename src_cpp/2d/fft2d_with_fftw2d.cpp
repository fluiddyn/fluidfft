

#include <iostream>
using namespace std;

#include <stdlib.h>

#include <sys/time.h>

#include <complex.h>
#include <fftw3.h>

#include <fft2d_with_fftw2d.h>


FFT2DWithFFTW2D::FFT2DWithFFTW2D(int argN0, int argN1):
  BaseFFT2D::BaseFFT2D(argN0, argN1)
{
  struct timeval start_time, end_time;
  myreal total_usecs;

  this->_init();
  
  /* y corresponds to dim 0 in physical space */
  /* x corresponds to dim 1 in physical space */
  ny = N0;
  nx = N1;
  
  nX0 = N0;
  nX1 = N1;
  nX0loc = nX0;
  nXyloc = nX0loc;

  nKx = nx/2+1;
  nKxloc = nKx;
  nKy = ny;

  /* This 2D fft is NOT transposed */
  nK0 = nKy;
  nK0loc = nK0;
  nK1 = nKx;

  coef_norm = N0*N1;

  flags = FFTW_MEASURE;
/*    flags = FFTW_ESTIMATE;*/
/*    flags = FFTW_PATIENT;*/
#ifdef SINGLE_PREC
  arrayX = fftwf_alloc_real(nX0 * nX1);
  arrayK = fftwf_alloc_complex(nK0 * nK1);
  gettimeofday(&start_time, NULL);

  plan_r2c = fftwf_plan_dft_r2c_2d(
      N0, N1, arrayX, arrayK, flags);

  plan_c2r = fftwf_plan_dft_c2r_2d(
      N0, N1, arrayK, arrayX, flags);
#else
  arrayX = fftw_alloc_real(nX0 * nX1);
  arrayK = fftw_alloc_complex(nK0 * nK1);
  gettimeofday(&start_time, NULL);

  plan_r2c = fftw_plan_dft_r2c_2d(
      N0, N1, arrayX, arrayK, flags);

  plan_c2r = fftw_plan_dft_c2r_2d(
      N0, N1, arrayK, arrayX, flags);
#endif

  gettimeofday(&end_time, NULL);

  total_usecs = (end_time.tv_sec-start_time.tv_sec) +
    (end_time.tv_usec-start_time.tv_usec)/1000000.;

  if (rank == 0)
    printf("Initialization (%s) done in %f s\n",
	   this->get_classname(), total_usecs);
}

void FFT2DWithFFTW2D::destroy(void)
{
  // cout << "Object is being destroyed" << endl;
#ifdef SINGLE_PREC
  fftwf_destroy_plan(plan_r2c);
  fftwf_destroy_plan(plan_c2r);
  fftwf_free(arrayX);
  fftwf_free(arrayK);
#else
  fftw_destroy_plan(plan_r2c);
  fftw_destroy_plan(plan_c2r);
  fftw_free(arrayX);
  fftw_free(arrayK);
#endif
}


FFT2DWithFFTW2D::~FFT2DWithFFTW2D(void)
{
}


char const* FFT2DWithFFTW2D::get_classname()
{ return "FFT2DWithFFTW2D";}


myreal FFT2DWithFFTW2D::compute_energy_from_X(myreal* fieldX)
{
  int ii;
  myreal energy = 0;

  for (ii=0; ii<nX0loc * nX1; ii++)
    energy += pow(fieldX[ii], 2);

  return energy / 2 /coef_norm;
}


myreal FFT2DWithFFTW2D::compute_energy_from_K(mycomplex* fieldK)
{
  int i0, i1;
  myreal energy = 0;
  myreal energy_tmp = 0;

  // modes i1 = iKx = 0
  i1 = 0;
  for (i0=0; i0<nK0; i0++)
    energy_tmp += pow(cabs(fieldK[i0 * nK1]), 2);
  
  energy = energy_tmp/2;

  // modes i1 = iKx = last = nK1 - 1
  i1 = nK1 - 1;
  energy_tmp = 0.;
  for (i0=0; i0<nK0; i0++)
    energy_tmp += pow(cabs(fieldK[i1 + i0 * nK1]), 2);

  if (nX1 % 2 == 0)
    energy_tmp = energy_tmp/2;

  energy += energy_tmp;

  // other modes
  for (i0=0; i0<nK0loc; i0++)
    for (i1=1; i1<nK1-1; i1++)
      energy += pow(cabs(fieldK[i1 + i0 * nK1]), 2);

  return energy;
}


myreal FFT2DWithFFTW2D::compute_mean_from_X(myreal* fieldX)
{
  myreal mean = 0.;
  int ii;

  for (ii=0; ii<nX0loc*nX1; ii++)
    mean += fieldX[ii];

  return mean / coef_norm;
}


myreal FFT2DWithFFTW2D::compute_mean_from_K(mycomplex* fieldK)
{
  myreal mean = creal(fieldK[0]);
  return mean;
}


void FFT2DWithFFTW2D::fft(myreal *fieldX, mycomplex *fieldK)
{
  int ii;
  // cout << "FFT2DWithFFTW2D::fft" << endl;

  /*use memcpy(void * destination, void * source, size_t bytes); */
  memcpy(arrayX, fieldX, nX0*nX1*sizeof(myreal));
#ifdef SINGLE_PREC
  fftwf_execute(plan_r2c);
#else  
  fftw_execute(plan_r2c);
#endif  
  for (ii=0; ii<nK0loc*nK1; ii++)
    fieldK[ii]  = arrayK[ii]/coef_norm;
}


void FFT2DWithFFTW2D::ifft(mycomplex *fieldK, myreal *fieldX)
{
  // cout << "FFT2DWithFFTW2D::ifft" << endl;
  memcpy(arrayK, fieldK, nK0*nK1*sizeof(mycomplex));
#ifdef SINGLE_PREC
  fftwf_execute(plan_c2r);
#else
  fftw_execute(plan_c2r);
#endif
  memcpy(fieldX, arrayX, nX0*nX1*sizeof(myreal));
}


void FFT2DWithFFTW2D::init_array_X_random(myreal* &fieldX)
{
  int ii;
  this->alloc_array_X(fieldX);

  for (ii = 0; ii < nX0*nX1; ++ii)
    fieldX[ii] = (myreal)rand() / RAND_MAX;
}
