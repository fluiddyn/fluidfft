

#include <iostream>
using namespace std;

#include <stdlib.h>

#include <sys/time.h>
#include <fft2d_with_cufft.h>



//  KERNEL CUDA
// Complex scale
static __device__ __host__ inline dcomplex ComplexScale(dcomplex a, real_cu s)
{
  dcomplex c;
  c.x = s * a.x;
  c.y = s * a.y;
  return c;
}

__global__ void vectorNorm(const real_cu norm, dcomplex *A, int numElements)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < numElements)
  {
    A[i] = ComplexScale(A[i], norm);
  }
}

////////////////// FIN KERNEL CUDA

FFT2DWithCUFFT::FFT2DWithCUFFT(int argN0, int argN1):
  BaseFFT2D::BaseFFT2D(argN0, argN1)
{
  struct timeval start_time, end_time;
  real_cu total_usecs;
  
  this->_init();

 /* y corresponds to dim 0 in physical space */
  /* x corresponds to dim 1 in physical space */
  ny = N0;
  nx = N1;

  nX0 = N0;
  nX0loc = nX0;
  nX1 = N1;
  nX1loc = nX1;
  

  nKx = nx/2+1;
  nKxloc = nKx;
  nKy = ny;
  
  /* This 2D fft is NOT transposed */
  nK0 = nKy;
  nK0loc = nK0;
  nK1 = nKx;
  nK1loc = nK1;
  
  coef_norm = N0*N1;


  mem_sizer = sizeof(real_cu) * N0 * N1 ;//taille de arrayX
  int new_size = nK0 * nK1 ;
  mem_size = 2 * sizeof(real_cu) * new_size ;//taille de arrayK

  gettimeofday(&start_time, NULL);
  // Allocate device memory for signal
  checkCudaErrors(cudaMalloc((void **)&data, mem_size));
  checkCudaErrors(cudaMalloc((void **)&datar, mem_sizer));

  // CUFFT plan
#ifdef SINGLE_PREC
  checkCudaErrors(cufftPlan2d(&plan, nX0, nX1, CUFFT_R2C));
  checkCudaErrors(cufftPlan2d(&plan1, nX0, nX1, CUFFT_C2R));
#else
  checkCudaErrors(cufftPlan2d(&plan, nX0, nX1, CUFFT_D2Z));
  checkCudaErrors(cufftPlan2d(&plan1, nX0, nX1, CUFFT_Z2D));
#endif

  gettimeofday(&end_time, NULL);

  total_usecs = (end_time.tv_sec-start_time.tv_sec) +
    (end_time.tv_usec-start_time.tv_usec)/1000000.;

  if (rank == 0)
    printf("Initialization (%s) done in %f s\n",
        this->get_classname(), total_usecs);
}


void FFT2DWithCUFFT::destroy(void)
{
  // cout << "Object is being destroyed" << endl;
cudaFree(data);
cudaFree(datar);
cufftDestroy(plan);
cufftDestroy(plan1);
}


FFT2DWithCUFFT::~FFT2DWithCUFFT(void)
{
}


char const* FFT2DWithCUFFT::get_classname()
{ return "FFT2DWithCUFFT";}


real_cu FFT2DWithCUFFT::compute_energy_from_X(real_cu* fieldX)
{
  int ii,jj;
  real_cu energy = 0.;
  real_cu energy1;

  for (ii=0; ii<nX0; ii++)
    {
    energy1=0.;
    for (jj=0; jj<nX1; jj++)
      {
      energy1 += pow(fieldX[ii*nX1+jj], 2);
      }
    energy += energy1 / nX1;
    }
  //cout << "energyX=" << energy / nX0 / 2 << endl;

  return energy / nX0 / 2;
}


#ifdef SINGLE_PREC
real_cu FFT2DWithCUFFT::compute_energy_from_K(fftwf_complex* fieldK)
#else
real_cu FFT2DWithCUFFT::compute_energy_from_K(fftw_complex* fieldK)
#endif
{
  int i0, i1;
  double energy = 0;
  double energy0 = 0;

  // modes i1_seq = iKx = last = nK1 - 1
  i1 = nK1 - 1;
  for (i0=0; i0<nK0; i0++)
    energy += (double) pow(cabs(fieldK[i1 + i0*nK1]), 2);//we must divide by 2 ==> after

    energy *= 0.5;//divide by 2!!!

  // other modes
  for (i0=0; i0<nK0; i0++)
    for (i1=1; i1<nK1-1; i1++)
        energy += (double) pow(cabs(fieldK[i1 + i0*nK1]), 2);
    
  // modes i1_seq = iKx = 0
  i1 = 0;
  for (i0=0; i0<nK0; i0++)
    energy0 += (double) pow(cabs(fieldK[i0*nK1]), 2);//we must divide by 2 ==> after

  energy += energy0*0.5;

  //cout << "energyK=" << energy<<  endl;
  return (real_cu) energy;
}


real_cu FFT2DWithCUFFT::compute_mean_from_X(real_cu* fieldX)
{
  real_cu mean,mean1;
  int ii,jj;
  mean=0.;

  for (ii=0; ii<nX0; ii++)
    {
    mean1=0.;
    for (jj=0; jj<nX1; jj++)
      {
      mean1 += fieldX[ii*nX1+jj];
      }
    mean += mean1 / nX1;
    }
  return mean / nX0;
}


#ifdef SINGLE_PREC
real_cu FFT2DWithCUFFT::compute_mean_from_K(fftwf_complex* fieldK)
#else
real_cu FFT2DWithCUFFT::compute_mean_from_K(fftw_complex* fieldK)
#endif
{
  real_cu mean;
  mean = creal(fieldK[0]);

  return mean;
}


#ifdef SINGLE_PREC
void FFT2DWithCUFFT::fft(real_cu *fieldX, fftwf_complex *fieldK)
#else
void FFT2DWithCUFFT::fft(real_cu *fieldX, fftw_complex *fieldK)
#endif
{
  
  
  // cout << "FFT2DWithCUFFT::fft" << endl;
  // Copy host memory to device
  checkCudaErrors(cudaMemcpy(datar, fieldX, mem_sizer, cudaMemcpyHostToDevice));

  
  // Transform signal and kernel
  //printf("Transforming signal cufftExecD2Z\n");
#ifdef SINGLE_PREC
  checkCudaErrors(cufftExecR2C(plan, (cufftReal *)datar, (cufftComplex *)data));
#else
  checkCudaErrors(cufftExecD2Z(plan, (cufftDoubleReal *)datar, (cufftDoubleComplex *)data));
#endif

  
  // Launch the Vector Norm CUDA Kernel
  real_cu norm = 1./coef_norm;
  //  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
  int threadsPerBlock = 256;
  int blocksPerGrid =(nK0 * nK1 + threadsPerBlock - 1) / threadsPerBlock;
  vectorNorm<<<blocksPerGrid, threadsPerBlock>>>(norm, data, nK0 * nK1 );
  

  // Copy host device to memory
  checkCudaErrors(cudaMemcpy(fieldK, data, mem_size, cudaMemcpyDeviceToHost));


}


#ifdef SINGLE_PREC
void FFT2DWithCUFFT::ifft(fftwf_complex *fieldK, real_cu *fieldX)
#else
void FFT2DWithCUFFT::ifft(fftw_complex *fieldK, real_cu *fieldX)
#endif
{

  //cout << "FFT2DWithCUFFT::ifft" << endl;
  // Copy host memory to device
  checkCudaErrors(cudaMemcpy(data, fieldK, mem_size, cudaMemcpyHostToDevice));


  // FFT on DEVICE
#ifdef SINGLE_PREC
  checkCudaErrors(cufftExecC2R(plan1, (cufftComplex *)data, (cufftReal *)datar));
#else
  checkCudaErrors(cufftExecZ2D(plan1, (cufftDoubleComplex *)data, (cufftDoubleReal *)datar));
#endif

  
  // Copy host device to memory
  checkCudaErrors(cudaMemcpy(fieldX, datar, mem_sizer, cudaMemcpyDeviceToHost));

}


void FFT2DWithCUFFT::init_array_X_random(real_cu* &fieldX)
{
  int ii;
  this->alloc_array_X(fieldX);

  for (ii = 0; ii < nX0*nX1; ++ii)
    fieldX[ii] = (real_cu)rand() / RAND_MAX;
}

