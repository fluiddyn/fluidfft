

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
  nX1loc = N1;
  nX2loc = N2;

  nKx = nx/2;
  nKxloc = (nKx)/nb_proc;
  nKy = ny;
  nKz = nz;

  if (((nX0loc == 0) || (nKxloc == 0)) and (rank == 0))
  {
    cout << "Warning: number of mpi processus not coherent with dimension of the problem!" << endl;
  }

  /* This 3D fft is transposed */
  is_transposed = 1;
  nK0 = nKx;
  nK0loc = nKxloc;
  nK1 = N1;
  nK1loc = N1;
  nK2 = N0;
  nK2loc = N0;
//  coef_norm = N0*N1*N2;

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
      reinterpret_cast<mycomplex_fftw*>(arrayK_pR), NULL,
      N1*nX0loc, 1,
      flags);
    
  plan_c2r = fftw_plan_many_dft_c2r(
      1, &N2, howmany,
      reinterpret_cast<mycomplex_fftw*>(arrayK_pR), NULL,
      N1*nX0loc, 1,
      arrayX, NULL,
      1, N2,
      flags);
  howmany = nX0loc*(nKx+1);
  sign = FFTW_FORWARD;
  plan_c2c1_fwd = fftw_plan_many_dft(
      1, &N1, howmany,
      reinterpret_cast<mycomplex_fftw*>(arrayK_pR), NULL,
      1, N1,
      reinterpret_cast<mycomplex_fftw*>(arrayK_pR), NULL,
      1, N1,
      sign, flags);
    
  sign = FFTW_BACKWARD;
  plan_c2c1_bwd = fftw_plan_many_dft(
      1, &N1, howmany,
      reinterpret_cast<mycomplex_fftw*>(arrayK_pR), NULL,
      1, N1,
      reinterpret_cast<mycomplex_fftw*>(arrayK_pR), NULL,
      1, N1,
      sign, flags);

  howmany = nKxloc*N1;
  sign = FFTW_FORWARD;
  plan_c2c_fwd = fftw_plan_many_dft(
      1, &N0, howmany,
      reinterpret_cast<mycomplex_fftw*>(arrayK_pC), &N0,
      istride, N0,
      reinterpret_cast<mycomplex_fftw*>(arrayK_pC), &N0,
      ostride, N0,
      sign, flags);

  sign = FFTW_BACKWARD;
  plan_c2c_bwd = fftw_plan_many_dft(
      1, &N0, howmany,
      reinterpret_cast<mycomplex_fftw*>(arrayK_pC), &N0,
      istride, N0,
      reinterpret_cast<mycomplex_fftw*>(arrayK_pC), &N0,
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


char const* FFT3DMPIWithFFTW1D::get_classname()
{ return "FFT3DMPIWithFFTW1D";}


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
      energy_tmp += square_abs(fieldK[i1*nK2 + i2]);
  
  if (rank == 0)  // i.e. if iKx == 0
    energy_loc = energy_tmp/2.;
  else
    energy_loc = energy_tmp;

  // other modes
  for (i0=1; i0<nK0loc; i0++)
    for (i1=0; i1<nK1; i1++)
      for (i2=0; i2<nK2; i2++)
        energy_loc += square_abs(fieldK[i2 + i1*nK2 + i0*nK1*nK2]);

  MPI_Allreduce(&energy_loc, &energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  // cout << "energy" << energy << endl;

  return energy;
}


myreal FFT3DMPIWithFFTW1D::sum_wavenumbers_double(myreal* fieldK)
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
    sum_loc = sum_tmp/2.;
  else
    sum_loc = sum_tmp;

  // other modes
  for (i0=1; i0<nK0loc; i0++)
    for (i1=0; i1<nK1; i1++)
      for (i2=0; i2<nK2; i2++)
        sum_loc += fieldK[i2 + i1*nK2 + i0*nK1*nK2];

  MPI_Allreduce(&sum_loc, &sum_tot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  // cout << "mean= " << sum_tot << endl;  

  return sum_tot*2.;
}


void FFT3DMPIWithFFTW1D::sum_wavenumbers_complex(
    mycomplex* fieldK, mycomplex* result)
{
  int i0, i1, i2;
  mycomplex sum_loc = 0;
  mycomplex sum_tmp = 0;
  mycomplex sum_tot;

  // modes i0 = iKx = 0
  i0 = 0;
  for (i1=0; i1<nK1; i1++)
    for (i2=0; i2<nK2; i2++)
      sum_tmp += fieldK[i2 + i1*nK2];
  
  if (rank == 0)  // i.e. if iKx == 0
    sum_loc = sum_tmp/2.;
  else
    sum_loc = sum_tmp;

  // other modes
  for (i0=1; i0<nK0loc; i0++)
    for (i1=0; i1<nK1; i1++)
      for (i2=0; i2<nK2; i2++)
        sum_loc += fieldK[i2 + i1*nK2 + i0*nK1*nK2];

  MPI_Allreduce(&sum_loc, &sum_tot, 1, MPI_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
  // cout << "mean= " << sum_tot << endl;  

  *result = sum_tot*2.;
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


void FFT3DMPIWithFFTW1D::get_dimX_K(int *d0, int *d1, int *d2)
{
  *d0 = 2;
  *d1 = 1;
  *d2 = 0;
}


bool FFT3DMPIWithFFTW1D::are_parameters_bad()
{
  if ((N0/nb_proc == 0) || (N2/2/nb_proc == 0))
    {
      if (rank == 0)
        cout << "bad parameters N0 or N2" << endl;
      return 1;
    }
  return 0;
}

