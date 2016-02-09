

#include <iostream>
using namespace std;

#include <stdlib.h>

#include <sys/time.h>

#include <complex.h>
#include <fftw3-mpi.h>

#ifdef OMP
#include <omp.h>
#endif

#include <fft3dmpi_with_fftwmpi3d.h>

FFT3DMPIWithFFTWMPI3D::FFT3DMPIWithFFTWMPI3D(int argN0, int argN1, int argN2):
  BaseFFT3DMPI::BaseFFT3DMPI(argN0, argN1, argN2)
{
  struct timeval start_time, end_time;
  myreal total_usecs;
  ptrdiff_t local_nX0;//, local_X0_start;
  ptrdiff_t local_nK0;

  this->_init();
#ifdef SINGLE_PREC
#ifdef OMP
  fftwf_init_threads();
#endif
  fftwf_mpi_init();
  /* get local data size and allocate */
  alloc_local = fftwf_mpi_local_size_3d_transposed(
      N0, N1, N2/2+1, MPI_COMM_WORLD,
      &local_nX0, &local_X0_start,
      &local_nK0, &local_K0_start);
#else
#ifdef OMP
  fftw_init_threads();
#endif
  fftw_mpi_init();
  /* get local data size and allocate */
  alloc_local = fftw_mpi_local_size_3d_transposed(
      N0, N1, N2/2+1, MPI_COMM_WORLD,
      &local_nX0, &local_X0_start,
      &local_nK0, &local_K0_start);
#endif
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

#ifdef SINGLE_PREC
  arrayX = fftwf_alloc_real(2 * alloc_local);
  arrayK = fftwf_alloc_complex(alloc_local);

  gettimeofday(&start_time, NULL);
#ifdef OMP
fftwf_plan_with_nthreads(omp_get_max_threads());
#endif
  plan_r2c = fftwf_mpi_plan_dft_r2c_3d(
      N0, N1, N2, arrayX, arrayK, MPI_COMM_WORLD,
      flags|FFTW_MPI_TRANSPOSED_OUT);

  plan_c2r = fftwf_mpi_plan_dft_c2r_3d(
      N0, N1, N2, arrayK, arrayX, MPI_COMM_WORLD,
      flags|FFTW_MPI_TRANSPOSED_IN);
#else
  arrayX = fftw_alloc_real(2 * alloc_local);
  arrayK = fftw_alloc_complex(alloc_local);

  gettimeofday(&start_time, NULL);
#ifdef OMP
fftw_plan_with_nthreads(omp_get_max_threads());
#endif
  plan_r2c = fftw_mpi_plan_dft_r2c_3d(
      N0, N1, N2, arrayX, arrayK, MPI_COMM_WORLD,
      flags|FFTW_MPI_TRANSPOSED_OUT);

  plan_c2r = fftw_mpi_plan_dft_c2r_3d(
      N0, N1, N2, arrayK, arrayX, MPI_COMM_WORLD,
      flags|FFTW_MPI_TRANSPOSED_IN);
#endif
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
#ifdef SINGLE_PREC
  fftwf_destroy_plan(plan_r2c);
  fftwf_destroy_plan(plan_c2r);
  fftwf_free(arrayX);
  fftwf_free(arrayK);
  fftwf_mpi_cleanup();
#ifdef OMP
  fftwf_cleanup_threads();
#endif
#else
  fftw_destroy_plan(plan_r2c);
  fftw_destroy_plan(plan_c2r);
  fftw_free(arrayX);
  fftw_free(arrayK);
  fftw_mpi_cleanup();
#ifdef OMP
  fftw_cleanup_threads();
#endif
#endif
}


FFT3DMPIWithFFTWMPI3D::~FFT3DMPIWithFFTWMPI3D(void)
{
}


char const* FFT3DMPIWithFFTWMPI3D::get_classname()
{ return "FFT3DMPIWithFFTWMPI3D";}


myreal FFT3DMPIWithFFTWMPI3D::compute_energy_from_X(myreal* fieldX)
{
  int ii;
  double energy_loc = 0;
  double energy;

  for (ii=0; ii<nX0loc*nX1*nX2; ii++)
	energy_loc += (double) pow(fieldX[ii], 2);

  MPI_Allreduce(&energy_loc, &energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  return (myreal) energy / 2 /coef_norm;
}

myreal FFT3DMPIWithFFTWMPI3D::compute_energy_from_K(mycomplex* fieldK)
{
  int i0, i1, i2;
  double energy_tmp = 0;
  double energy_loc;
  double energy;

  // modes i2 = iKx = 0
  i2 = 0;
  for (i0=0; i0<nK0loc; i0++)
    for (i1=0; i1<nK1; i1++)
      energy_tmp += (double) pow(cabs(fieldK[(i1 + i0 * nK1) * nK2]), 2);

  energy_loc = energy_tmp/2;

  // modes i2 = iKx = last = nK2 - 1
  i2 = nK2 - 1;
  energy_tmp = 0;
  for (i0=0; i0<nK0loc; i0++)
    for (i1=0; i1<nK1; i1++)
      energy_tmp += (double) pow(cabs(fieldK[i2 + (i1 + i0 * nK1) * nK2]), 2);

  if (N2%2 == 0)
    energy_loc += energy_tmp/2;
  else
    energy_loc += energy_tmp;
  
  // other modes
  for (i0=0; i0<nK0loc; i0++)
    for (i1=0; i1<nK1; i1++)
      for (i2=1; i2<nK2-1; i2++)
	energy_loc += (double) pow(cabs(fieldK[i2 + (i1 + i0 * nK1) * nK2]), 2);

  MPI_Allreduce(&energy_loc, &energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  return (myreal) energy;
}


myreal FFT3DMPIWithFFTWMPI3D::sum_wavenumbers_double(myreal* fieldK)
{
  int i0, i1, i2;
  double energy_tmp = 0;
  double energy_loc, energy;

  // modes i2 = iKx = 0
  i2 = 0;
  for (i0=0; i0<nK0loc; i0++)
    for (i1=0; i1<nK1; i1++)
      energy_tmp += (double) fieldK[(i1 + i0 * nK1) * nK2];

  energy_loc = energy_tmp/2;

  // modes i2 = iKx = last = nK2 - 1
  i2 = nK2 - 1;
  energy_tmp = 0;
  for (i0=0; i0<nK0loc; i0++)
    for (i1=0; i1<nK1; i1++)
      energy_tmp += (double) fieldK[i2 + (i1 + i0 * nK1) * nK2];

  if (N2%2 == 0)
    energy_loc += energy_tmp/2;
  else
    energy_loc += energy_tmp;
  
  // other modes
  for (i0=0; i0<nK0loc; i0++)
    for (i1=0; i1<nK1; i1++)
      for (i2=1; i2<nK2-1; i2++)
	energy_loc += (double) fieldK[i2 + (i1 + i0 * nK1) * nK2];

  MPI_Allreduce(&energy_loc, &energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  return (myreal) energy;
}

void FFT3DMPIWithFFTWMPI3D::sum_wavenumbers_complex(
    mycomplex* fieldK, mycomplex* result)
{
  int i0, i1, i2;
  mycomplex energy_tmp = 0;
  mycomplex energy_loc, energy;
  // modes i2 = iKx = 0
  i2 = 0;
  for (i0=0; i0<nK0loc; i0++)
    for (i1=0; i1<nK1; i1++)
      energy_tmp += fieldK[(i1 + i0 * nK1) * nK2];

  energy_loc = energy_tmp/2;

  // modes i2 = iKx = last = nK2 - 1
  i2 = nK2 - 1;
  energy_tmp = 0;
  for (i0=0; i0<nK0loc; i0++)
    for (i1=0; i1<nK1; i1++)
      energy_tmp += fieldK[i2 + (i1 + i0 * nK1) * nK2];

  if (N2%2 == 0)
    energy_loc += energy_tmp/2;
  else
    energy_loc += energy_tmp;
  
  // other modes
  for (i0=0; i0<nK0loc; i0++)
    for (i1=0; i1<nK1; i1++)
      for (i2=1; i2<nK2-1; i2++)
	energy_loc += fieldK[i2 + (i1 + i0 * nK1) * nK2];

  MPI_Allreduce(&energy_loc, &energy, 1, MPI_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
//ERREUR PTET LA... COMPLEX OU DOUBLE COMPLEX???
  *result = energy;
}



myreal FFT3DMPIWithFFTWMPI3D::compute_mean_from_X(myreal* fieldX)
{
  double mean, local_mean;
  int ii;
  local_mean=0.;

  for (ii=0; ii<nX0loc*nX1*nX2; ii++)
    local_mean += fieldX[ii];

  MPI_Allreduce(&local_mean, &mean, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  return (myreal) mean / coef_norm;
}

myreal FFT3DMPIWithFFTWMPI3D::compute_mean_from_K(mycomplex* fieldK)
{
  double mean, local_mean;
  if (local_K0_start == 0)
    local_mean = (double) creal(fieldK[0]);
  else
    local_mean = 0.;

  MPI_Allreduce(&local_mean, &mean, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  return (myreal) mean;
}

void FFT3DMPIWithFFTWMPI3D::fft(myreal *fieldX, mycomplex *fieldK)
{
  int i0, i1, i2;
  // cout << "FFT3DMPIWithFFTWMPI3D::fft" << endl;

  for (i0=0; i0<nX0loc; i0++)
    for (i1=0; i1<nX1; i1++)
      for (i2=0; i2<nX2; i2++)
	arrayX[i2 + (i1 + i0*nX1)*nX2_pad] = fieldX[i2 + (i1 + i0*nX1)*nX2];

#ifdef SINGLE_PREC
  fftwf_execute(plan_r2c);
#else
  fftw_execute(plan_r2c);
#endif

  for (i0=0; i0<nK0loc; i0++)
    for (i1=0; i1<nK1; i1++)
      for (i2=0; i2<nK2; i2++)
	fieldK[i2 + (i1 + i0*nK1)*nK2]  =
	  arrayK[i2 + (i1 + i0*nK1)*nK2]/coef_norm;
}

void FFT3DMPIWithFFTWMPI3D::ifft(mycomplex *fieldK, myreal *fieldX)
{
  int i0, i1, i2;
  // cout << "FFT3DMPIWithFFTWMPI3D::ifft" << endl;
  memcpy(arrayK, fieldK, alloc_local*sizeof(mycomplex));
#ifdef SINGLE_PREC
  fftwf_execute(plan_c2r);
#else
  fftw_execute(plan_c2r);
#endif

  for (i0=0; i0<nX0loc; i0++)
    for (i1=0; i1<nX1; i1++)
      for (i2=0; i2<nX2; i2++)
        fieldX[i2 + (i1 + i0*nX1)*nX2] = arrayX[i2 + (i1 + i0*nX1)*nX2_pad];
}


void FFT3DMPIWithFFTWMPI3D::init_array_X_random(myreal* &fieldX)
{
  int ii;
  this->alloc_array_X(fieldX);

  for (ii = 0; ii < nX0loc*nX1*nX2; ++ii)
    fieldX[ii] = (myreal)rand() / RAND_MAX;
}


void FFT3DMPIWithFFTWMPI3D::get_dimX_K(int *d0, int *d1, int *d2)
{
  *d0 = 1;
  *d1 = 0;
  *d2 = 2;
}

void FFT3DMPIWithFFTWMPI3D::get_seq_indices_first_K(int *i0, int *i1)
{
  *i0 = local_K0_start;
  *i1 = 0;
}
