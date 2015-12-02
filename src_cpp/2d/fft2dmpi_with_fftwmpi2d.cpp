

#include <iostream>
using namespace std;

#include <stdlib.h>

#include <sys/time.h>

#include <complex.h>
#include <fftw3-mpi.h>

#include <fft2dmpi_with_fftwmpi2d.h>


FFT2DMPIWithFFTWMPI2D::FFT2DMPIWithFFTWMPI2D(int argN0, int argN1):
  BaseFFT2DMPI::BaseFFT2DMPI(argN0, argN1)
{
  struct timeval start_time, end_time;
  double total_usecs;
  ptrdiff_t local_nX0;//, local_X0_start;
  ptrdiff_t local_nK0;

  this->_init();

  fftw_mpi_init();

  /* get local data size and allocate */
  alloc_local = fftw_mpi_local_size_2d_transposed(N0, N1/2+1, MPI_COMM_WORLD,
                                                  &local_nX0, &local_X0_start,
                                                  &local_nK0, &local_K0_start);

  /* y corresponds to dim 0 in physical space */
  /* x corresponds to dim 1 in physical space */
  ny = N0;
  nx = N1;
  
  nX0 = N0;
  nX1 = N1;
  nX0loc = local_nX0;
  nXyloc = nX0loc;
  nX1_pad = N1 + 2;

  nKx = nx/2+1;
  nKxloc = local_nK0;
  nKy = ny;

  /* This 2D fft is transposed */
  nK0 = nKx;
  nK0loc = nKxloc;
  nK1 = nKy;

  coef_norm = N0*N1;

  flags = FFTW_MEASURE;
/*    flags = FFTW_ESTIMATE;*/
/*    flags = FFTW_PATIENT;*/

  arrayX = fftw_alloc_real(2 * alloc_local);
  arrayK = fftw_alloc_complex(alloc_local);

  gettimeofday(&start_time, NULL);

  plan_r2c = fftw_mpi_plan_dft_r2c_2d(
      N0, N1, arrayX,
      arrayK, MPI_COMM_WORLD,
      flags|FFTW_MPI_TRANSPOSED_OUT);

  plan_c2r = fftw_mpi_plan_dft_c2r_2d(
      N0, N1,
      arrayK, arrayX,
      MPI_COMM_WORLD,
      flags|FFTW_MPI_TRANSPOSED_IN);

  gettimeofday(&end_time, NULL);

  total_usecs = (end_time.tv_sec-start_time.tv_sec) +
    (end_time.tv_usec-start_time.tv_usec)/1000000.;

  if (rank == 0)
    printf("Initialization (%s) done in %f s\n",
	   this->get_classname(), total_usecs);

}

void FFT2DMPIWithFFTWMPI2D::destroy(void)
{
  // cout << "Object is being destroyed" << endl;
  fftw_destroy_plan(plan_r2c);
  fftw_destroy_plan(plan_c2r);
  fftw_free(arrayX);
  fftw_free(arrayK);
}


FFT2DMPIWithFFTWMPI2D::~FFT2DMPIWithFFTWMPI2D(void)
{
}


char const* FFT2DMPIWithFFTWMPI2D::get_classname()
{ return "FFT2DMPIWithFFTWMPI2D";}


double FFT2DMPIWithFFTWMPI2D::compute_energy_from_X(double* fieldX)
{
  int i0, i1;
  double energy_loc = 0;
  double energy;

  for (i0=0; i0<nX0loc; i0++)
    for (i1=0; i1<nX1; i1++)
      energy_loc += pow(fieldX[i1 + i0*nX1], 2);

  MPI_Allreduce(&energy_loc, &energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  return energy / 2 /coef_norm;
}


double FFT2DMPIWithFFTWMPI2D::compute_energy_from_K(fftw_complex* fieldK)
{
  int i0, i1, i_tmp;
  double energy_loc = 0;
  double energy_tmp = 0;
  double energy;

  // modes i0 = iKx = 0
  i0 = 0;
  for (i1=0; i1<nK1; i1++)
    energy_tmp += pow(cabs(fieldK[i1]), 2);
  
  if (local_K0_start == 0)  // i.e. if iKx == 0
    energy_loc = energy_tmp/2;
  else
    energy_loc = energy_tmp;

  // modes i0 = iKx = last = nK0loc - 1
  i0 = nK0loc - 1;
  energy_tmp = 0.;
  i_tmp = i0 * nK1;
  for (i1=0; i1<nK1; i1++)
    energy_tmp += pow(cabs(fieldK[i1 + i_tmp]), 2);

  if (rank == nb_proc - 1)
    energy_loc += energy_tmp/2;
  else
    energy_loc += energy_tmp;

  // other modes
  for (i0=1; i0<nK0loc-1; i0++)
    for (i1=0; i1<nK1; i1++)
      energy_loc += pow(cabs(fieldK[i1 + i0*nK1]), 2);

  MPI_Allreduce(&energy_loc, &energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  return energy;
}


double FFT2DMPIWithFFTWMPI2D::compute_mean_from_X(double* fieldX)
{
  double mean, local_mean;
  int ii;
  local_mean=0.;

  for (ii=0; ii<nX0loc*nX1; ii++)
    local_mean += fieldX[ii];

  MPI_Allreduce(&local_mean, &mean, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  return mean / coef_norm;
}


double FFT2DMPIWithFFTWMPI2D::compute_mean_from_K(fftw_complex* fieldK)
{
  double mean, local_mean;
  if (local_K0_start == 0)
    local_mean = creal(fieldK[0]);
  else
    local_mean = 0.;

  MPI_Allreduce(&local_mean, &mean, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  return mean;
}


void FFT2DMPIWithFFTWMPI2D::fft(double *fieldX, fftw_complex *fieldK)
{
  int i0, i1;
  // cout << "FFT2DMPIWithFFTWMPI2D::fft" << endl;

  for (i0=0; i0<nX0loc; i0++)
    for (i1=0; i1<nX1; i1++)
      arrayX[i1 + i0*nX1_pad] = fieldX[i1 + i0*nX1];
  
  fftw_execute(plan_r2c);

  for (i0=0; i0<nK0loc; i0++)
    for (i1=0; i1<nK1; i1++)
      fieldK[i1 + i0*nK1]  = arrayK[i1 + i0*nK1]/coef_norm;
}


void FFT2DMPIWithFFTWMPI2D::ifft(fftw_complex *fieldK, double *fieldX)
{
  int i0, i1;
  // cout << "FFT2DMPIWithFFTWMPI2D::ifft" << endl;
  memcpy(arrayK, fieldK, alloc_local*sizeof(fftw_complex));
  fftw_execute(plan_c2r);

  for (i0=0; i0<nX0loc; i0++)
    for (i1=0; i1<nX1; i1++)
      fieldX[i1 + i0*nX1] = arrayX[i1 + i0*nX1_pad];
}


void FFT2DMPIWithFFTWMPI2D::init_array_X_random(double* &fieldX)
{
  int ii;
  this->alloc_array_X(fieldX);
  // for (i0=0; i0<nX0loc; i0++)
  //   for (i1=0; i1<nX1; i1++)
  //     fieldX[i1 + i0*nX1] = 1.3425243;
  // if (local_X0_start==0) 
  // {
  //   for (i0=0; i0<nX0loc; i0++)
  //     for (i1=0; i1<nX1; i1++)
  //       fieldX[i1 + i0*nX1] = 1111.3425243;
  // }

  for (ii = 0; ii < nX0loc*nX1; ++ii)
    fieldX[ii] = (double)rand() / RAND_MAX;
}


