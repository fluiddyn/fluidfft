

#include <iostream>
using namespace std;

#include <stdlib.h>

#include <sys/time.h>

#include <complex.h>
#include <fftw3-mpi.h>

#include <fft3dmpi_with_fftwmpi3d.h>


FFT3DMPIWithFFTWMPI3D::FFT3DMPIWithFFTWMPI3D(int argN0, int argN1, int argN2):
  BaseFFT3DMPI::BaseFFT3DMPI(argN0, argN1, argN2)
{
  struct timeval start_time, end_time;
  double total_usecs;
  ptrdiff_t local_nX0;//, local_X0_start;
  ptrdiff_t local_nK0;

  this->_init();

  fftw_mpi_init();

  /* get local data size and allocate */
  alloc_local = fftw_mpi_local_size_3d_transposed(
      N0, N1, N2/2+1, MPI_COMM_WORLD,
      &local_nX0, &local_X0_start,
      &local_nK0, &local_K0_start);

  /* in physical space: */
  /* z corresponds to dim 0 */
  /* y corresponds to dim 1 */
  /* x corresponds to dim 2 */
  nz = N0;
  ny = N1;
  nx = N2;

  nX0 = N0;
  nX1 = N1;
  nX2 = N2;

  nX0loc = local_nX0;
  nXzloc = nX0loc;
  nX1loc = nX1;

  nX2_pad = 2*(N2/2 + 1);

  /* This 3D fft is transposed */
  /* in Fourier space: */
  /* ky corresponds to dim 0 */
  /* kz corresponds to dim 1 */
  /* kx corresponds to dim 2 */
  nKx = nx/2+1;
  nK2 = nKx;

  nKy = ny;
  nK0 = nKy;
  nK0loc = local_nK0;
  nKyloc = nK0loc;

  nK1 = N0;
  nK1loc = nK1;

  coef_norm = N0*N1*N2;

  flags = FFTW_MEASURE;
/*    flags = FFTW_ESTIMATE;*/
/*    flags = FFTW_PATIENT;*/

  arrayX = fftw_alloc_real(2 * alloc_local);
  arrayK = fftw_alloc_complex(alloc_local);

  gettimeofday(&start_time, NULL);

  plan_r2c = fftw_mpi_plan_dft_r2c_3d(
      N0, N1, N2, arrayX, arrayK, MPI_COMM_WORLD,
      flags|FFTW_MPI_TRANSPOSED_OUT);

  plan_c2r = fftw_mpi_plan_dft_c2r_3d(
      N0, N1, N2, arrayK, arrayX, MPI_COMM_WORLD,
      flags|FFTW_MPI_TRANSPOSED_IN);

  gettimeofday(&end_time, NULL);

  total_usecs = (end_time.tv_sec-start_time.tv_sec) +
    (end_time.tv_usec-start_time.tv_usec)/1000000.;

  if (rank == 0)
    printf("Initialization (%s) done in %f s\n",
	   this->get_classname(), total_usecs);

}

void FFT3DMPIWithFFTWMPI3D::destroy(void)
{
  // cout << "Object is being destroyed" << endl;
  fftw_destroy_plan(plan_r2c);
  fftw_destroy_plan(plan_c2r);
  fftw_free(arrayX);
  fftw_free(arrayK);
}


FFT3DMPIWithFFTWMPI3D::~FFT3DMPIWithFFTWMPI3D(void)
{
}


char const* FFT3DMPIWithFFTWMPI3D::get_classname()
{ return "FFT3DMPIWithFFTWMPI3D";}


double FFT3DMPIWithFFTWMPI3D::compute_energy_from_X(double* fieldX)
{
  int ii;
  double energy_loc = 0;
  double energy;

  for (ii=0; ii<nX0loc*nX1*nX2; ii++)
	energy_loc += pow(fieldX[ii], 2);

  MPI_Allreduce(&energy_loc, &energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  return energy / 2 /coef_norm;
}


double FFT3DMPIWithFFTWMPI3D::compute_energy_from_K(fftw_complex* fieldK)
{
  int i0, i1, i2;
  double energy_tmp = 0;
  double energy_loc;
  double energy;

  // modes i2 = iKx = 0
  i2 = 0;
  for (i0=0; i0<nK0loc; i0++)
    for (i1=0; i1<nK1; i1++)
      energy_tmp += pow(cabs(fieldK[(i1 + i0 * nK1) * nK2]), 2);

  energy_loc = energy_tmp/2;

  // modes i2 = iKx = last = nK2 - 1
  i2 = nK2 - 1;
  energy_tmp = 0;
  for (i0=0; i0<nK0loc; i0++)
    for (i1=0; i1<nK1; i1++)
      energy_tmp += pow(cabs(fieldK[i2 + (i1 + i0 * nK1) * nK2]), 2);

  if (N2%2 == 0)
    energy_loc += energy_tmp/2;
  else
    energy_loc += energy_tmp;
  
  // other modes
  for (i0=0; i0<nK0loc; i0++)
    for (i1=0; i1<nK1; i1++)
      for (i2=1; i2<nK2-1; i2++)
	energy_loc += pow(cabs(fieldK[i2 + (i1 + i0 * nK1) * nK2]), 2);

  MPI_Allreduce(&energy_loc, &energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  return energy;
}


double FFT3DMPIWithFFTWMPI3D::compute_mean_from_X(double* fieldX)
{
  double mean, local_mean;
  int ii;
  local_mean=0.;

  for (ii=0; ii<nX0loc*nX1*nX2; ii++)
    local_mean += fieldX[ii];

  MPI_Allreduce(&local_mean, &mean, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  return mean / coef_norm;
}


double FFT3DMPIWithFFTWMPI3D::compute_mean_from_K(fftw_complex* fieldK)
{
  double mean, local_mean;
  if (local_K0_start == 0)
    local_mean = creal(fieldK[0]);
  else
    local_mean = 0.;

  MPI_Allreduce(&local_mean, &mean, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  return mean;
}


void FFT3DMPIWithFFTWMPI3D::fft(double *fieldX, fftw_complex *fieldK)
{
  int i0, i1, i2;
  // cout << "FFT3DMPIWithFFTWMPI3D::fft" << endl;

  for (i0=0; i0<nX0loc; i0++)
    for (i1=0; i1<nX1; i1++)
      for (i2=0; i2<nX2; i2++)
	arrayX[i2 + (i1 + i0*nX1)*nX2_pad] = fieldX[i2 + (i1 + i0*nX1)*nX2];

  fftw_execute(plan_r2c);

  for (i0=0; i0<nK0loc; i0++)
    for (i1=0; i1<nK1; i1++)
      for (i2=0; i2<nK2; i2++)
	fieldK[i2 + (i1 + i0*nK1)*nK2]  =
	  arrayK[i2 + (i1 + i0*nK1)*nK2]/coef_norm;
}


void FFT3DMPIWithFFTWMPI3D::ifft(fftw_complex *fieldK, double *fieldX)
{
  int i0, i1, i2;
  // cout << "FFT3DMPIWithFFTWMPI3D::ifft" << endl;
  memcpy(arrayK, fieldK, alloc_local*sizeof(fftw_complex));
  fftw_execute(plan_c2r);

  for (i0=0; i0<nX0loc; i0++)
    for (i1=0; i1<nX1; i1++)
      for (i2=0; i2<nX2; i2++)
        fieldX[i2 + (i1 + i0*nX1)*nX2] = arrayX[i2 + (i1 + i0*nX1)*nX2_pad];
}


void FFT3DMPIWithFFTWMPI3D::init_array_X_random(double* &fieldX)
{
  int ii;
  this->alloc_array_X(fieldX);

  for (ii = 0; ii < nX0loc*nX1*nX2; ++ii)
    fieldX[ii] = (double)rand() / RAND_MAX;
}
