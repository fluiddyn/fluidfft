

#include <iostream>
using namespace std;

#include <stdlib.h>

#include <sys/time.h>

#include <complex.h>
#include <fftw3.h>

#include <mpi.h>

#include <fft3dmpi_with_fftw1d.h>


FFT3DMPIWithFFTW1D::FFT3DMPIWithFFTW1D(int argN0, int argN1, int argN2):
  BaseFFT3DMPI::BaseFFT3DMPI(argN0, argN1, argN2)
{
  struct timeval start_time, end_time;
  myreal total_usecs;
  int iX0;
  int istride = 1, ostride = 1;
  int howmany, sign;
  MPI_Datatype MPI_type_complex;

  this->_init();

  /* z corresponds to dim 0 in physical space */
  /* y corresponds to dim 1 in physical space */
  /* x corresponds to dim 2 in physical space */  
  nz = N0;
  ny = N1;
  nx = N2;
  
  nX0 = N0;
  nX1 = N1;
  nX2 = N2;
  nX0loc = N0/nb_proc;
  nXzloc = nX0loc;

  nKx = nx/2;
  nKxloc = (nKx)/nb_proc;
  nKy = ny;
  nKz = nz;

  /* This 3D fft is transposed */
  is_transposed = 1;
  nK0 = nKx;
  nK0loc = nKxloc;
  nK1 = N1;
  nK2 = N0;
  coef_norm = N0*N1*N2;

  local_X0_start = rank * nX0loc;
  local_K0_start = rank * nK0loc;

  flags = FFTW_MEASURE;
/*    flags = FFTW_ESTIMATE;*/
/*    flags = FFTW_PATIENT;*/

  arrayX    = (myreal*) fftw_malloc(sizeof(myreal)*nX0loc*N1*N2);
  arrayK_pR = (mycomplex*) fftw_malloc(sizeof(mycomplex)
                                            *nX0loc*(nKx+1)*nKy);
  arrayK_pC = (mycomplex*) fftw_malloc(sizeof(mycomplex)*nKxloc*nK2*N1);

  gettimeofday(&start_time, NULL);

  howmany = nX0loc*N1;

  plan_r2c = fftw_plan_many_dft_r2c(
      1, &N2, howmany,
      arrayX, NULL,
      1, N2,
      arrayK_pR, NULL,
      N1*nX0loc, 1,
      flags);
    
  plan_c2r = fftw_plan_many_dft_c2r(
      1, &N2, howmany,
      arrayK_pR, NULL,
      N1*nX0loc, 1,
      arrayX, NULL,
      1, N2,
      flags);
  howmany = nX0loc*(nKx+1);
  sign = FFTW_FORWARD;
  plan_c2c1_fwd = fftw_plan_many_dft(
      1, &N1, howmany,
      arrayK_pR, NULL,
      1, N1,
      arrayK_pR, NULL,
      1, N1,
      sign, flags);
    
  sign = FFTW_BACKWARD;
  plan_c2c1_bwd = fftw_plan_many_dft(
      1, &N1, howmany,
      arrayK_pR, NULL,
      1, N1,
      arrayK_pR, NULL,
      1, N1,
      sign, flags);

  howmany = nKxloc*N1;
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

  for (iX0=0;iX0<nX0loc;iX0++)
    arrayK_pR[iX0*(nKx+1)+nKx] = 0.;


  MPI_Type_contiguous( 2, MPI_DOUBLE, &MPI_type_complex );
  MPI_Type_commit( &MPI_type_complex );
  
  MPI_Type_vector(nX0loc, 1, nKy, 
                  MPI_type_complex, &(MPI_type_column));
  MPI_Type_commit( &(MPI_type_column) );

  MPI_Type_create_hvector(N1, 1, 16, 
                  MPI_type_column, &(MPI_type_block1));
  MPI_Type_commit( &(MPI_type_block1) );

  MPI_Type_create_hvector(nKxloc, 1, N1*nX0loc*16, 
                  MPI_type_block1, &(MPI_type_block2));
  MPI_Type_commit( &(MPI_type_block2) );

  MPI_Type_vector(nKxloc*N1, nX0loc, N0, 
                  MPI_type_complex, &(MPI_type_block));
  MPI_Type_create_resized(MPI_type_block, 0, 
                          nX0loc*sizeof(mycomplex), 
                          &(MPI_type_block));
  MPI_Type_commit( &(MPI_type_block) );


}

void FFT3DMPIWithFFTW1D::destroy(void)
{
  // cout << "Object is being destroyed" << endl;
  fftw_destroy_plan(plan_r2c);
  fftw_destroy_plan(plan_c2c_fwd);
  fftw_destroy_plan(plan_c2c_bwd);
  fftw_destroy_plan(plan_c2c1_fwd);
  fftw_destroy_plan(plan_c2c1_bwd);
  fftw_destroy_plan(plan_c2r);
  fftw_free(arrayX);
  fftw_free(arrayK_pR);
  fftw_free(arrayK_pC);
  MPI_Type_free(&(MPI_type_column));
  MPI_Type_free(&(MPI_type_block));
  MPI_Type_free(&(MPI_type_block1));
  MPI_Type_free(&(MPI_type_block2));
}


FFT3DMPIWithFFTW1D::~FFT3DMPIWithFFTW1D(void)
{
  // cout << "Object is being deleted" << endl;
}


char const* FFT3DMPIWithFFTW1D::get_classname()
{ return "FFT3DMPIWithFFTW1D";}


myreal FFT3DMPIWithFFTW1D::compute_energy_from_X(myreal* fieldX)
{
  int i0, i1, i2;
  myreal energy_loc = 0;
  myreal energy;

  for (i0=0; i0<nX0loc; i0++)
    for (i1=0; i1<nX1; i1++)
      for (i2=0; i2<nX2; i2++)
        energy_loc += pow(fieldX[i2 + i1*nX2 + i0*nX1*nX2], 2);

  MPI_Allreduce(&energy_loc, &energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  // cout << "energyX" << energy / 2 /coef_norm << endl;

  return energy / 2 /coef_norm;
}


myreal FFT3DMPIWithFFTW1D::compute_energy_from_K(mycomplex* fieldK)
{
  int i0, i1, i2;
  myreal energy_loc = 0;
  myreal energy_tmp = 0;
  myreal energy;

  // modes i0 = iKx = 0
  i0 = 0;
  for (i1=0; i1<nK1; i1++)
    for (i2=0; i2<nK2; i2++)
      energy_tmp += pow(cabs(fieldK[i1*nK2 + i2]), 2);
  
  if (rank == 0)  // i.e. if iKx == 0
    energy_loc = energy_tmp/2;
  else
    energy_loc = energy_tmp;

  // other modes
  for (i0=1; i0<nK0loc; i0++)
    for (i1=0; i1<nK1; i1++)
      for (i2=0; i2<nK2; i2++)
        energy_loc += pow(cabs(fieldK[i2 + i1*nK2 + i0*nK1*nK2]), 2);

  MPI_Allreduce(&energy_loc, &energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  // cout << "energy" << energy << endl;

  return energy;
}


myreal FFT3DMPIWithFFTW1D::sum_wavenumbers(myreal* fieldK)
{
  int i0, i1, i2;
  myreal sum_loc = 0;
  myreal sum_tmp = 0;
  myreal sum_tot;

  // modes i0 = iKx = 0
  i0 = 0;
  for (i1=0; i1<nK1; i1++)
    for (i2=0; i2<nK2; i2++)
      sum_tmp += fieldK[i2 + i1*nK2];
  
  if (rank == 0)  // i.e. if iKx == 0
    sum_loc = sum_tmp/2;
  else
    sum_loc = sum_tmp;

  // other modes
  for (i0=1; i0<nK0loc; i0++)
    for (i1=0; i1<nK1; i1++)
      for (i2=0; i2<nK2; i2++)
        sum_loc += fieldK[i2 + i1*nK2 + i0*nK1*nK2];

  MPI_Allreduce(&sum_loc, &sum_tot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  // cout << "mean= " << sum_tot << endl;  

  return sum_tot;
}




myreal FFT3DMPIWithFFTW1D::compute_mean_from_X(myreal* fieldX)
{
  myreal mean, local_mean;
  int ii;
  local_mean=0.;

  for (ii=0; ii<nX0loc*nX1*nX2; ii++)
    local_mean += fieldX[ii];

  MPI_Allreduce(&local_mean, &mean, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);


 // cout << "mean="<<mean / coef_norm<<endl; 
  return mean / coef_norm;
}


myreal FFT3DMPIWithFFTW1D::compute_mean_from_K(mycomplex* fieldK)
{
  myreal mean;
  if (rank == 0)
    mean = creal(fieldK[0]);

  MPI_Bcast(&mean, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
 // cout << "meanK="<<mean<<endl; 
  return mean;
}


void FFT3DMPIWithFFTW1D::fft(myreal *fieldX, mycomplex *fieldK)
{
  int ii;
  // cout << "FFT3DMPIWithFFTW1D::fft" << endl;
  /*use memcpy(void * destination, void * source, size_t bytes); */
  memcpy(arrayX, fieldX, nX0loc*nX1*nX2*sizeof(myreal));

  fftw_execute(plan_r2c);
  fftw_execute(plan_c2c1_fwd);

  MPI_Alltoall(arrayK_pR, 1, MPI_type_block2,
               arrayK_pC, 1, MPI_type_block,
               MPI_COMM_WORLD);

  fftw_execute(plan_c2c_fwd);

  for (ii=0; ii<nKxloc*nKy*nKz; ii++)
    fieldK[ii]  = arrayK_pC[ii]/coef_norm;
}


void FFT3DMPIWithFFTW1D::ifft(mycomplex *fieldK, myreal *fieldX)
{
  int ii;
  // cout << "FFT3DMPIWithFFTW1D::ifft" << endl;
  memcpy(arrayK_pC, fieldK, nKxloc*nKy*nKz*sizeof(mycomplex));
  fftw_execute(plan_c2c_bwd);

  MPI_Alltoall(arrayK_pC, 1, MPI_type_block,
	       arrayK_pR, 1, MPI_type_block2,
	       MPI_COMM_WORLD);
  
  /*These modes (nx/2+1=N1/2+1) have to be settled to zero*/
  for (ii = 0; ii < N1*nX0loc; ++ii) 
    arrayK_pR[N1*nX0loc*nKx + ii] = 0.;

  fftw_execute(plan_c2c1_bwd);
  fftw_execute(plan_c2r);
  memcpy(fieldX,arrayX, nX0loc*nX1*nX2*sizeof(myreal));
}


void FFT3DMPIWithFFTW1D::init_array_X_random(myreal* &fieldX)
{
  int ii;
  this->alloc_array_X(fieldX);

  // cout << "init fieldX" << endl;
  for (ii = 0; ii < nX0loc*nX1*nX2; ++ii)
    fieldX[ii] = (myreal)rand() / RAND_MAX;
}


int FFT3DMPIWithFFTW1D::get_local_size_K()
{
  // cout << "FFT3DMPIWithFFTW1D::get_local_size_K" << endl;
  return nKxloc * nKy * nKz;
}


int FFT3DMPIWithFFTW1D::get_local_size_X()
{
  // cout << "FFT3DMPIWithFFTW1D::get_local_size_X" << endl;
  return nX0loc * nX1 * nX2;
}
