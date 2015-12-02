

#include <iostream>
using namespace std;

#include <stdlib.h>
#include <string.h>

#include <sys/time.h>

#include <complex.h>
#include <fftw3.h>

#include <fft2d_with_fftw1d.h>


FFT2DWithFFTW1D::FFT2DWithFFTW1D(int argN0, int argN1):
  BaseFFT2D::BaseFFT2D(argN0, argN1)
{
  struct timeval start_time, end_time;
  double total_usecs;
  int iX0;
  int istride = 1, ostride = 1;
  int howmany, sign;

  this->_init();
  
  /* y corresponds to dim 0 in physical space */
  /* x corresponds to dim 1 in physical space */
  ny = N0;
  nx = N1;
  
  nX0 = N0;
  nX1 = N1;
  nX0loc = nX0;
  nXyloc = nX0loc;

  nKx = nx/2;
  nKxloc = nKx;
  nKy = ny;

  /* This 2D fft is transposed */
  nK0 = nKx;
  nK0loc = nK0;
  nK1 = nKy;

  coef_norm = N0*N1;

  flags = FFTW_MEASURE;
/*    flags = FFTW_ESTIMATE;*/
/*    flags = FFTW_PATIENT;*/

  arrayX    = (double*) fftw_malloc(sizeof(double)*nX0loc * nX1);
  arrayK_pR = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) *
                                          nX0loc * (nKx+1));
  arrayK_pC = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nKxloc * N0);

  gettimeofday(&start_time, NULL);

  howmany = nX0loc;
  plan_r2c = fftw_plan_many_dft_r2c(
      1, &N1, howmany,
      arrayX, NULL,
      istride, N1,
      arrayK_pR, NULL,
      ostride, nKx+1,
      flags);
    
  plan_c2r = fftw_plan_many_dft_c2r(
      1, &N1, howmany,
      arrayK_pR, NULL,
      istride, nKx+1,
      arrayX, NULL,
      ostride, N1,
      flags);

  howmany = nKxloc;
  sign = FFTW_FORWARD;
  plan_c2c_fwd = fftw_plan_many_dft(
      1, &N0, howmany,
      arrayK_pC, &N0,
      istride, N0,
      arrayK_pC, &N0,
      ostride, N0,
      sign, flags);

  sign = FFTW_BACKWARD;
  plan_c2c_bwd = fftw_plan_many_dft(
      1, &N0, howmany,
      arrayK_pC, &N0,
      istride, N0,
      arrayK_pC, &N0,
      ostride, N0,
      sign, flags);

  gettimeofday(&end_time, NULL);

  total_usecs = (end_time.tv_sec-start_time.tv_sec) +
    (end_time.tv_usec-start_time.tv_usec)/1000000.;

  if (rank == 0)
    printf("Initialization (%s) done in %f s\n",
	   this->get_classname(), total_usecs);

  for (iX0=0; iX0<nX0loc; iX0++)
    arrayK_pR[iX0*(nKx+1)+nKx] = 0.;

}

void FFT2DWithFFTW1D::destroy(void)
{
  // cout << "Object is being destroyed" << endl;
  fftw_destroy_plan(plan_r2c);
  fftw_destroy_plan(plan_c2c_fwd);
  fftw_destroy_plan(plan_c2c_bwd);
  fftw_destroy_plan(plan_c2r);
  fftw_free(arrayX);
  fftw_free(arrayK_pR);
  fftw_free(arrayK_pC);
}

FFT2DWithFFTW1D::~FFT2DWithFFTW1D(void)
{
  // cout << "Object is being deleted" << endl;
}

char const* FFT2DWithFFTW1D::get_classname()
{ return "FFT2DWithFFTW1D";}


double FFT2DWithFFTW1D::compute_energy_from_X(double* fieldX)
{
  int ii;
  double energy = 0;

  for (ii=0; ii<nX0loc * nX1; ii++)
    energy += pow(fieldX[ii], 2);

  return energy / 2 /coef_norm;
}

double FFT2DWithFFTW1D::compute_energy_from_K(fftw_complex* fieldK)
{
  int i0, i1;
  double energy = 0;
  double energy_tmp = 0;

  // modes i0 = iKx = 0
  i0 = 0;
  for (i1=0; i1<nK1; i1++)
    energy_tmp += pow(cabs(fieldK[i1]), 2);
  
  energy = energy_tmp/2;

  // other modes
  for (i0=1; i0<nK0loc; i0++)
    for (i1=0; i1<nK1; i1++)
      energy += pow(cabs(fieldK[i1 + i0*nK1]), 2);

  return energy;
}

double FFT2DWithFFTW1D::compute_mean_from_X(double* fieldX)
{
  double mean = 0.;
  int ii;

  for (ii=0; ii<nX0loc*nX1; ii++)
    mean += fieldX[ii];

  return mean / coef_norm;
}

double FFT2DWithFFTW1D::compute_mean_from_K(fftw_complex* fieldK)
{
  double mean = creal(fieldK[0]);
  return mean;
}

void FFT2DWithFFTW1D::fft(double *fieldX, fftw_complex *fieldK)
{
  int ii, jj;
  // cout << "FFT2DWithFFTW1D::fft" << endl;
  /*use memcpy(void * destination, void * source, size_t bytes); */
  memcpy(arrayX, fieldX, nX0loc*nX1*sizeof(double));

  fftw_execute(plan_r2c);

  /*    second step: transpose...*/
  for (ii = 0; ii < nX0; ++ii)
    for (jj = 0; jj < nKx; ++jj)
      arrayK_pC[jj*nX0+ii] = 
	arrayK_pR[ii*(nKx+1)+jj];

  fftw_execute(plan_c2c_fwd);

  for (ii=0; ii<nKxloc*nKy; ii++)
    fieldK[ii]  = arrayK_pC[ii]/coef_norm;
  
}

void FFT2DWithFFTW1D::ifft(fftw_complex *fieldK, double *fieldX)
{
  int ii, jj;
  // cout << "FFT2DWithFFTW1D::ifft" << endl;
  memcpy(arrayK_pC, fieldK, nKxloc*nKy*sizeof(fftw_complex));
  fftw_execute(plan_c2c_bwd);

  /* second step: transpose...*/
  for (ii = 0; ii < nKx; ++ii)
    for (jj = 0; jj < nX0; ++jj)
      arrayK_pR[jj*(nKx+1)+ii] =
	arrayK_pC[ii*nX0+jj];
  
  /*These modes (nx/2+1=N1/2+1) have to be settled to zero*/
  for (ii = 0; ii < nX0loc; ++ii) 
    arrayK_pR[ii*(nKx+1) + nKx] = 0.;

  fftw_execute(plan_c2r);
  memcpy(fieldX,arrayX, nX0loc*nX1*sizeof(double));
}

void FFT2DWithFFTW1D::init_array_X_random(double* &fieldX)
{
  int ii;
  this->alloc_array_X(fieldX);

  for (ii = 0; ii < nX0*nX1; ++ii)
    fieldX[ii] = (double)rand() / RAND_MAX;
}
