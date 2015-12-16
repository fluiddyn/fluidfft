

#include <iostream>
using namespace std;

#include <stdlib.h>

#include <sys/time.h>

#include <fft3dmpi_with_p3dfft.h>


FFT3DMPIWithP3DFFT::FFT3DMPIWithP3DFFT(int argN0, int argN1, int argN2):
  BaseFFT3DMPI::BaseFFT3DMPI(argN0, argN1, argN2)
{
  struct timeval start_time, end_time;
  double total_usecs;
  int irank;
  
  int ndim, conf;
  int memsize[3];
  int istart[3],isize[3],iend[3];
  int fstart[3],fsize[3],fend[3];

  
  this->_init();

  ndim=2;
  nXz = N0;
  nXy = N1;
  nXx = N2;

  nX0 = N0;
  nX1 = N1;
  nX2 = N2;


  if(ndim == 1) {
    nprocmesh[0] = 1; nprocmesh[1] = nb_proc;
  }
  else if(ndim == 2) {
    nprocmesh[0] = 2;
    nprocmesh[1] = nb_proc/2;
    }

  gettimeofday(&start_time, NULL);

  if(rank == 0)
    printf("Using processor grid %d x %d\n",nprocmesh[0],nprocmesh[1]);

  /* Initialize P3DFFT */
  Cp3dfft_setup(nprocmesh,N0,N1,N2,MPI_Comm_c2f(MPI_COMM_WORLD),nX0,nX1,nX2,0,memsize);
  /* Get dimensions for input array - real numbers, X-pencil shape.
   *    *       Note that we are following the Fortran ordering, i.e. 
   *       *             the dimension  with stride-1 is X. */
  conf = 1;
  Cp3dfft_get_dims(istart,iend,isize,conf);
  /* Get dimensions for output array - complex numbers, Z-pencil shape.
   *    *       Stride-1 dimension could be X or Z, depending on how the library 
   *       *             was compiled (stride1 option) */
  conf = 2;
  Cp3dfft_get_dims(fstart,fend,fsize,conf);

  arrayX = (double *) malloc(sizeof(double) * isize[0]*isize[1]*isize[2]);
  arrayK = (double *) malloc(sizeof(double) * fsize[0]*fsize[1]*fsize[2]*2);




  /* in physical space: */
  /* z corresponds to dim 0 */
  /* y corresponds to dim 1 */
  /* x corresponds to dim 2 */
  nX0loc = isize[0];
  nXxloc = nX0loc;
  nX1loc = isize[1];
  nXyloc = nX1loc;
  nX2loc = isize[2];
  nXzloc = nX2loc;


  /* This 3D fft is transposed */
  /* in Fourier space: */
  /* ky corresponds to dim 0 */
  /* kx corresponds to dim 1 */
  /* kz corresponds to dim 2 */
  nKx = nXx; // /2+1;
  nKy = nXy;
  nKz = nXz;

  nK0 = nKz;
  nK1 = nKy;
  nK2 = nKx;

  // Warning: order as in X space!
  nKzloc = fsize[0];
  nKyloc = fsize[1];
  nKxloc = fsize[2];

 /* nKyloc = fsize[0];
  nKxloc = fsize[1];
  nKzloc = fsize[2];
*/
  nK0loc = nKyloc;
  nK1loc = nKxloc;
  nK2loc = nKzloc;

  nK0loc = nKzloc;
  nK1loc = nKyloc;
  nK2loc = nKxloc;
  MPI_Barrier(MPI_COMM_WORLD);
  fflush(stdout);
  
  MPI_Barrier(MPI_COMM_WORLD);
  local_X0_start = istart[0];
  local_X1_start = istart[1];
  // Warning: order as in X space!
  local_K0_start = fstart[0];
  local_K1_start = fstart[1];
  
  coef_norm = N0*N1*N2;

  gettimeofday(&end_time, NULL);

  total_usecs = (end_time.tv_sec-start_time.tv_sec) +
    (end_time.tv_usec-start_time.tv_usec)/1000000.;

  if (rank == 0)
    printf("Initialization (%s) done in %f s\n",
	   this->get_classname(), total_usecs);
}

void FFT3DMPIWithP3DFFT::destroy(void)
{
//   cout << "Object is being destroyed" << endl;
   Cp3dfft_clean();
   free(arrayX);
   free(arrayK);
}


FFT3DMPIWithP3DFFT::~FFT3DMPIWithP3DFFT(void)
{
}


char const* FFT3DMPIWithP3DFFT::get_classname()
{ return "FFT3DMPIWithP3DFFT";}


double FFT3DMPIWithP3DFFT::compute_energy_from_X(double* fieldX)
{
  int ii;
  double energy_loc = 0;
  double energy;

  for (ii=0; ii<nX0loc*nX1loc*nX2loc; ii++)
	energy_loc += pow(fieldX[ii], 2);

  MPI_Allreduce(&energy_loc, &energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  return energy / 2 /coef_norm;

}


double FFT3DMPIWithP3DFFT::compute_energy_from_K(fftw_complex* fieldK)
{
  int i0, i1, i2;
  double energy_tmp = 0;
  double energy_loc;
  double energy;
  i0 = 0;
  energy_tmp=0.;
  for (i2=0; i2<nK2loc; i2++)
    for (i1=0; i1<nK1loc; i1++)
      energy_tmp+= pow(cabs(fieldK[i0 + (i1+i2*nK1loc)*nK0loc]), 2);

  if (local_K0_start == 1)
    energy_loc = energy_tmp /2.;
  else
    energy_loc = energy_tmp;

//  modes i1_seq iKx = last = nK1 - 1
    i0 = nK0loc - 1;
  energy_tmp = 0;
  for (i2=0; i2<nK2loc; i2++)
    for (i1=0; i1<nK1loc; i1++)
      energy_tmp += pow(cabs(fieldK[i0 + (i1 + i2*nK1loc)*nK0loc]), 2);

  if (local_K0_start + nK0loc -1== nK0/2+1)
    energy_loc += energy_tmp/2;
  else
    energy_loc += energy_tmp;

  for (i2=0; i2<nK2loc; i2++)
    for (i1=0; i1<nK1loc; i1++)
      for (i0=1; i0<nK0loc-1; i0++)
        energy_loc += pow(cabs(fieldK[i0 + (i1 + i2*nK1loc)*nK0loc]), 2);

  
  MPI_Allreduce(&energy_loc, &energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  return energy;
}


double FFT3DMPIWithP3DFFT::compute_mean_from_X(double* fieldX)
{
  double mean, local_mean;
  int ii;
  local_mean=0.;

  for (ii=0; ii < nX0loc * nX1loc * nX2loc ; ii++)
    local_mean += fieldX[ii];

  MPI_Allreduce(&local_mean, &mean, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  return mean / coef_norm;
}


double FFT3DMPIWithP3DFFT::compute_mean_from_K(fftw_complex* fieldK)
{
  double mean, local_mean;
 
  if (local_K0_start == 1 and local_K1_start == 1)
    local_mean = creal(fieldK[0]);
  else
    local_mean = 0.;

  MPI_Allreduce(&local_mean, &mean, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  return mean;
}


void FFT3DMPIWithP3DFFT::fft(double *fieldX, fftw_complex *fieldK)
{
  int i0, i1, i2;
  unsigned char op_f[]="fft";
  //cout << "FFT3DMPIWithP3DFFT::fft" << endl;
  double coef_normdiv = 1./coef_norm;

  memcpy(arrayX, fieldX, nX0loc*nX1loc*nX2loc*sizeof(double));
  Cp3dfft_ftran_r2c(arrayX,arrayK,op_f);
  memcpy(fieldK, arrayK, nK0loc*nK1loc*nK2loc*sizeof(fftw_complex));


  for (i0=0; i0<nK0loc*nK1loc*nK2loc; i0++)
    fieldK[i0]  *= coef_normdiv;
}


void FFT3DMPIWithP3DFFT::ifft(fftw_complex *fieldK, double *fieldX)
{
  unsigned char op_b[]="tff";
  //cout << "FFT3DMPIWithP3DFFT::ifft" << endl;
  memcpy(arrayK, fieldK, nK0loc*nK1loc*nK2loc*sizeof(fftw_complex));
  Cp3dfft_btran_c2r(arrayK,arrayX,op_b);
  memcpy(fieldX, arrayX, nX0loc*nX1loc*nX2loc*sizeof(double)); 
}


void FFT3DMPIWithP3DFFT::init_array_X_random(double* &fieldX)
{
  int ii;
  this->alloc_array_X(fieldX);

  for (ii = 0; ii < nX0loc*nX1loc*nX2loc; ++ii)
    fieldX[ii] = (double)rand() / RAND_MAX;
}
