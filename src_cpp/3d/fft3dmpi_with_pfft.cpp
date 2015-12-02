

#include <iostream>
using namespace std;

#include <stdlib.h>

#include <sys/time.h>

#include <complex.h>
#include <pfft.h>

#include <fft3dmpi_with_pfft.h>


FFT3DMPIWithPFFT::FFT3DMPIWithPFFT(int argN0, int argN1, int argN2):
  BaseFFT3DMPI::BaseFFT3DMPI(argN0, argN1, argN2)
{
  struct timeval start_time, end_time;
  double total_usecs;
  unsigned flag_fwd, flag_bck;
  int irank;
  
  this->_init();

  pfft_init();

  nprocmesh[0] = 2;
  nprocmesh[1] = nb_proc/2;

  /* Create two-dimensional process grid of size np[0] x np[1], if possible */
  if(pfft_create_procmesh_2d(MPI_COMM_WORLD, nprocmesh[0], nprocmesh[1],
			      &comm_cart_2d)){
    pfft_fprintf(MPI_COMM_WORLD, stderr,
		 "Error: This test file only works with %d processes.\n",
		 nprocmesh[0]*nprocmesh[1]);
    MPI_Finalize();
  }
  
  N[0] = N0;
  N[1] = N1;
  N[2] = N2;

  flag_fwd = PFFT_TRANSPOSED_OUT;
  flag_bck = PFFT_TRANSPOSED_IN;
  
  /* Get parameters of data distribution */
  alloc_local = pfft_local_size_dft_r2c_3d(
      N, comm_cart_2d, flag_fwd,
      local_ni, local_i_start, local_no, local_o_start);

  /* in physical space: */
  /* z corresponds to dim 0 */
  /* y corresponds to dim 1 */
  /* x corresponds to dim 2 */
  nXz = N0;
  nXy = N1;
  nXx = N2;

  nX0 = N0;
  nX1 = N1;
  nX2 = N2;

  nX0loc = local_ni[0];
  nXzloc = nX0loc;
  nX1loc = local_ni[1];
  nXyloc = nX1loc;
  nX2loc = local_ni[2];
  nXxloc = nX2loc;

  if (nX2loc != nX2)
    cout << "Warning: nX2loc != nX2\n";

  /* This 3D fft is transposed */
  /* in Fourier space: */
  /* ky corresponds to dim 0 */
  /* kx corresponds to dim 1 */
  /* kz corresponds to dim 2 */
  nKx = nXx/2+1;
  nKy = nXy;
  nKz = nXz;

  nK0 = nKy;
  nK1 = nKx;
  nK2 = nKz;

  // Warning: order as in X space!
  nKzloc = local_no[0];
  nKyloc = local_no[1];
  nKxloc = local_no[2];

  nK0loc = nKyloc;
  nK1loc = nKxloc;
  nK2loc = nKzloc;

  MPI_Barrier(MPI_COMM_WORLD);
  fflush(stdout);
  if (nK2loc != nK2)
    {
      cout << "Warning: nK2loc != nK2\n";
      for (irank=0; irank<nb_proc; irank++)
	{
	  MPI_Barrier(MPI_COMM_WORLD);
	  if (irank == rank)
	    printf("rank%d: nK2 = %d ; nK2loc: %d; nX0loc: %d ; local_ni[0] %zu\n",
		   rank, nK2, nK2loc, nX0loc, local_ni[0]);
	} 
    }
  
  MPI_Barrier(MPI_COMM_WORLD);
  local_X0_start = local_i_start[0];
  local_X1_start = local_i_start[1];
  // Warning: order as in X space!
  local_K0_start = local_o_start[1];
  local_K1_start = local_o_start[2];

  if (local_o_start[0] != 0)
    cout << "Warning: local_o_start[0] != 0" << endl;
  
  // if (alloc_local != nK0loc*nK1loc*nK2loc)
  //   cout << "Warning: alloc_local: " << alloc_local
  // 	 << " != nK0loc*nK1loc*nK2loc: " << nK0loc*nK1loc*nK2loc << endl;


  // for (irank=0; irank<nb_proc; irank++)
  //   {
  //     fflush(stdout);
  //     MPI_Barrier(MPI_COMM_WORLD);
  //     if (irank == rank)
  // 	printf("rank%2d: nX0: %d ; nX1: %d ; nX2: %d ; "
  // 	       "nK0: %d ; nK1: %d ; nK2: %d\n", rank,
  // 	       nX0, nX1, nX2, nK0, nK1, nK2);
  // 	printf("rank%2d: "
  // 	       "local_ni: %zu %zu %zu ; "
  // 	       "local_i_start: %zu %zu %zu ; "
  // 	       "local_no: %zu %zu %zu ; "
  // 	       "local_o_start: %zu %zu %zu ; \n"
  // 	       "        nK0loc: %d ; nK1loc: %d ; nK2loc: %d ;"
  // 	       " local_K0_start: %zu ; local_K1_start: %zu\n",
  // 	       rank,
  // 	       local_ni[0], local_ni[1], local_ni[2],
  // 	       local_i_start[0], local_i_start[1], local_i_start[2],
  // 	       local_no[0], local_no[1], local_no[2],
  // 	       local_o_start[0], local_o_start[1], local_o_start[2],
  // 	       nK0loc, nK1loc, nK2loc,
  // 	       local_K0_start, local_K1_start);
  //   }
  
  coef_norm = N0*N1*N2;

  flags = PFFT_MEASURE;
/*    flags = PFFT_ESTIMATE;*/
/*    flags = PFFT_PATIENT;*/

  arrayX = pfft_alloc_real(2 * alloc_local);
  arrayK = pfft_alloc_complex(alloc_local);

  gettimeofday(&start_time, NULL);

  plan_r2c = pfft_plan_dft_r2c_3d(
      N, arrayX, arrayK, comm_cart_2d, PFFT_FORWARD,
      flag_fwd | PFFT_MEASURE| PFFT_DESTROY_INPUT);

  plan_c2r = pfft_plan_dft_c2r_3d(
      N, arrayK, arrayX, comm_cart_2d, PFFT_BACKWARD,
      flag_bck | PFFT_MEASURE| PFFT_DESTROY_INPUT);

  gettimeofday(&end_time, NULL);

  total_usecs = (end_time.tv_sec-start_time.tv_sec) +
    (end_time.tv_usec-start_time.tv_usec)/1000000.;

  if (rank == 0)
    printf("Initialization (%s) done in %f s\n",
	   this->get_classname(), total_usecs);
}

void FFT3DMPIWithPFFT::destroy(void)
{
  // cout << "Object is being destroyed" << endl;
  pfft_destroy_plan(plan_r2c);
  pfft_destroy_plan(plan_c2r);
  MPI_Comm_free(&comm_cart_2d);
  pfft_free(arrayX);
  pfft_free(arrayK);
}


FFT3DMPIWithPFFT::~FFT3DMPIWithPFFT(void)
{
}


char const* FFT3DMPIWithPFFT::get_classname()
{ return "FFT3DMPIWithPFFT";}


double FFT3DMPIWithPFFT::compute_energy_from_X(double* fieldX)
{
  int ii;
  double energy_loc = 0;
  double energy;

  for (ii=0; ii<nX0loc*nX1loc*nX2; ii++)
	energy_loc += pow(fieldX[ii], 2);

  MPI_Allreduce(&energy_loc, &energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  return energy / 2 /coef_norm;
}


double FFT3DMPIWithPFFT::compute_energy_from_K(fftw_complex* fieldK)
{
  int i0, i1, i2;
  double energy_tmp = 0;
  double energy_loc;
  double energy;

  // modes i1_seq = iKx = 0
  i1 = 0;
  for (i0=0; i0<nK0loc; i0++)
    for (i2=0; i2<nK2; i2++)
      energy_tmp += pow(cabs(fieldK[i2 + (i0*nK1loc)*nK2]), 2);

  if (local_K1_start == 0)
    energy_loc = energy_tmp/2;
  else
    energy_loc = energy_tmp;

  // modes i1_seq = iKx = last = nK1 - 1
  i1 = nK1loc - 1;
  energy_tmp = 0;
  for (i0=0; i0<nK0loc; i0++)
    for (i2=0; i2<nK2; i2++)
      energy_tmp += pow(cabs(fieldK[i2 + (i1 + i0*nK1loc)*nK2]), 2);

  if (N1%2 == 0 and local_K1_start + nK1loc == nK1)
      energy_loc += energy_tmp/2;
  else
    energy_loc += energy_tmp;
  
  // other modes
  for (i0=0; i0<nK0loc; i0++)
    for (i1=1; i1<nK1loc-1; i1++)
      for (i2=0; i2<nK2; i2++)
	energy_loc += pow(cabs(fieldK[i2 + (i1 + i0*nK1loc)*nK2]), 2);

  MPI_Allreduce(&energy_loc, &energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  return energy;
}


double FFT3DMPIWithPFFT::compute_mean_from_X(double* fieldX)
{
  double mean, local_mean;
  int ii;
  local_mean=0.;

  for (ii=0; ii<nX0loc*nX1loc*nX2; ii++)
    local_mean += fieldX[ii];

  MPI_Allreduce(&local_mean, &mean, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  return mean / coef_norm;
}


double FFT3DMPIWithPFFT::compute_mean_from_K(fftw_complex* fieldK)
{
  double mean, local_mean;
  if (local_K0_start == 0 and local_K1_start == 0)
    local_mean = creal(fieldK[0]);
  else
    local_mean = 0.;

  MPI_Allreduce(&local_mean, &mean, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  return mean;
}


void FFT3DMPIWithPFFT::fft(double *fieldX, fftw_complex *fieldK)
{
  int i0, i1, i2;
  // cout << "FFT3DMPIWithPFFT::fft" << endl;

  memcpy(arrayX, fieldX, nX0loc*nX1loc*nX2*sizeof(double));

  pfft_execute(plan_r2c);
  
  for (i0=0; i0<nK0loc; i0++)
    for (i1=0; i1<nK1loc; i1++)
      for (i2=0; i2<nK2; i2++)
	fieldK[i2 + (i1 + i0*nK1loc)*nK2]  =
	  arrayK[i2 + (i1 + i0*nK1loc)*nK2]/coef_norm;
}


void FFT3DMPIWithPFFT::ifft(fftw_complex *fieldK, double *fieldX)
{
  // cout << "FFT3DMPIWithPFFT::ifft" << endl;
  memcpy(arrayK, fieldK, nK0loc*nK1loc*nK2*sizeof(fftw_complex));
  pfft_execute(plan_c2r);
  memcpy(fieldX, arrayX, nX0loc*nX1loc*nX2*sizeof(double));
}


void FFT3DMPIWithPFFT::init_array_X_random(double* &fieldX)
{
  int ii;
  this->alloc_array_X(fieldX);

  for (ii = 0; ii < nX0loc*nX1loc*nX2; ++ii)
    fieldX[ii] = (double)rand() / RAND_MAX;
}
