

#include <iostream>
using namespace std;

#include <stdlib.h>

#include <sys/time.h>

#include <complex.h>
#include <fftw3.h>

#include <fft3d_with_fftw3d.h>


FFT3DWithFFTW3D::FFT3DWithFFTW3D(int argN0, int argN1, int argN2):
  BaseFFT3D::BaseFFT3D(argN0, argN1, argN2)
{
  struct timeval start_time, end_time;
  double total_usecs;

  this->_init();

  /* y corresponds to dim 0 in physical space */
  /* y corresponds to dim 1 in physical space */
  /* x corresponds to dim 2 in physical space */
  nz = N0;
  ny = N1;
  nx = N2;
  
  nX0 = N0;
  nX1 = N1;
  nX2 = N2;
  nX0loc = nX0;
  nX1loc = nX1;

  nKx = nx/2+1;
  nKxloc = nKx;
  nKy = ny;
  nKz = nz;

  /* This 3D fft is NOT transposed */
  nK0 = nKz;
  nK0loc = nK0;
  nK1 = nKy;
  nK1loc = nK1;
  nK2 = nKx;
  
  coef_norm = N0*N1*N2;

  flags = FFTW_MEASURE;
/*    flags = FFTW_ESTIMATE;*/
/*    flags = FFTW_PATIENT;*/

  arrayX = fftw_alloc_real(nX0 * nX1 * nX2);
  arrayK = fftw_alloc_complex(nK0 * nK1 * nK2);

  gettimeofday(&start_time, NULL);

  plan_r2c = fftw_plan_dft_r2c_3d(
      N0, N1, N2, arrayX, arrayK, flags);

  plan_c2r = fftw_plan_dft_c2r_3d(
      N0, N1, N2, arrayK, arrayX, flags);

  gettimeofday(&end_time, NULL);

  total_usecs = (end_time.tv_sec-start_time.tv_sec) +
    (end_time.tv_usec-start_time.tv_usec)/1000000.;

  if (rank == 0)
    printf("Initialization (%s) done in %f s\n",
	   this->get_classname(), total_usecs);
}

void FFT3DWithFFTW3D::destroy(void)
{
  // cout << "Object is being destroyed" << endl;
  fftw_destroy_plan(plan_r2c);
  fftw_destroy_plan(plan_c2r);
  fftw_free(arrayX);
  fftw_free(arrayK);
}


FFT3DWithFFTW3D::~FFT3DWithFFTW3D(void)
{
}


char const* FFT3DWithFFTW3D::get_classname()
{ return "FFT3DWithFFTW3D";}


double FFT3DWithFFTW3D::compute_energy_from_X(double* fieldX)
{
  int ii;
  double energy = 0;

  for (ii=0; ii<nX0loc * nX1loc * nX2; ii++)
    energy += pow(fieldX[ii], 2);

  return energy / 2 /coef_norm;
}


double FFT3DWithFFTW3D::compute_energy_from_K(fftw_complex* fieldK)
{
  int i0, i1, i2;
  double energy = 0;
  double energy_tmp = 0;

  // modes i2 = iKx = 0
  i2 = 0;
  for (i0=0; i0<nK0; i0++)
    for (i1=0; i1<nK1; i1++)
      energy_tmp += pow(cabs(fieldK[(i1 + i0 * nK1) * nK2]), 2);
  
  energy = energy_tmp/2;

  // modes i2 = iKx = last = nK2 - 1
  i2 = nK2 - 1;
  energy_tmp = 0.;
  for (i0=0; i0<nK0; i0++)
    for (i1=0; i1<nK1; i1++)
      energy_tmp += pow(cabs(fieldK[i2 + (i1 + i0 * nK1) * nK2]), 2);

  if (N2%2 == 0)
    energy += energy_tmp/2;
  else
    energy += energy_tmp;

  // other modes
  for (i0=0; i0<nK0loc; i0++)
    for (i1=0; i1<nK1; i1++)
      for (i2=1; i2<nK2-1; i2++)
	energy += pow(cabs(fieldK[i2 + (i1 + i0 * nK1) * nK2]), 2);

  return energy;
}


double FFT3DWithFFTW3D::compute_mean_from_X(double* fieldX)
{
  double mean = 0.;
  int ii;

  for (ii=0; ii<nX0loc*nX1loc*nX2; ii++)
    mean += fieldX[ii];

  return mean / coef_norm;
}


double FFT3DWithFFTW3D::compute_mean_from_K(fftw_complex* fieldK)
{
  double mean = creal(fieldK[0]);
  return mean;
}


void FFT3DWithFFTW3D::fft(double *fieldX, fftw_complex *fieldK)
{
  int ii;
  // cout << "FFT3DWithFFTW3D::fft" << endl;

  /*use memcpy(void * destination, void * source, size_t bytes); */
  memcpy(arrayX, fieldX, nX0*nX1*nX2*sizeof(double));
  
  fftw_execute(plan_r2c);
  
  for (ii=0; ii<nK0loc*nK1loc*nK2; ii++)
    fieldK[ii]  = arrayK[ii]/coef_norm;
}


void FFT3DWithFFTW3D::ifft(fftw_complex *fieldK, double *fieldX)
{
  // cout << "FFT3DWithFFTW3D::ifft" << endl;
  memcpy(arrayK, fieldK, nK0*nK1*nK2*sizeof(fftw_complex));
  fftw_execute(plan_c2r);
  memcpy(fieldX, arrayX, nX0*nX1*nX2*sizeof(double));
}


void FFT3DWithFFTW3D::init_array_X_random(double* &fieldX)
{
  int ii;
  this->alloc_array_X(fieldX);

  for (ii = 0; ii < nX0*nX1*nX2; ++ii)
    fieldX[ii] = (double)rand() / RAND_MAX;
}
