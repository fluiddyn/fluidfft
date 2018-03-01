

#include <iostream>
using namespace std;

#include <stdlib.h>

#include <sys/time.h>
#include <fft3d_with_cufft.h>


//  KERNEL CUDA
// Complex scale
static __device__ __host__ inline dcomplex ComplexScale(dcomplex a, myreal s)
{
  dcomplex c;
  c.x = s * a.x;
  c.y = s * a.y;
  return c;
}

__global__ void vectorNorm(const myreal norm, dcomplex *A, int numElements)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < numElements)
  {
    A[i] = ComplexScale(A[i], norm);
  }
}

////////////////// FIN KERNEL CUDA

FFT3DWithCUFFT::FFT3DWithCUFFT(int argN0, int argN1, int argN2):
  BaseFFT3D::BaseFFT3D(argN0, argN1, argN2)
{
  struct timeval start_time, end_time;
  myreal total_usecs;
  
  this->_init();

  /* y corresponds to dim 0 in physical space */
  /* y corresponds to dim 1 in physical space */
  /* x corresponds to dim 2 in physical space */
  nz = N0;
  ny = N1;
  nx = N2;

  nX0 = N0;
  nX0loc = nX0;
  nX1 = N1;
  nX1loc = nX1;
  nX2 = N2;
  nX2loc = N2;

  nKx = nx/2+1;
  nKy = ny;
  nKz = nz;

  /* This 3D fft is NOT transposed */
  nK0 = nKz;
  nK0loc = nK0;
  nK1 = nKy;
  nK1loc = nK1;
  nK2 = nKx;
  nK2loc = nK2;

  mem_sizer = sizeof(myreal) * N0 * N1 * N2 ;//taille de arrayX
  int new_size = nK0 * nK1 * nK2 ;
  mem_size = 2 * sizeof(myreal) * new_size ;//taille de arrayK

  gettimeofday(&start_time, NULL);
  // Allocate device memory for signal
  checkCudaErrors(cudaMalloc((void **)&data, mem_size));
  checkCudaErrors(cudaMalloc((void **)&datar, mem_sizer));

  // CUFFT plan
#ifdef SINGLE_PREC
  checkCudaErrors(cufftPlan3d(&plan, nX0, nX1, nX2, CUFFT_R2C));
  checkCudaErrors(cufftPlan3d(&plan1, nX0, nX1, nX2, CUFFT_C2R));
#else
  checkCudaErrors(cufftPlan3d(&plan, nX0, nX1, nX2, CUFFT_D2Z));
  checkCudaErrors(cufftPlan3d(&plan1, nX0, nX1, nX2, CUFFT_Z2D));
#endif

  gettimeofday(&end_time, NULL);

  total_usecs = (end_time.tv_sec-start_time.tv_sec) +
    (end_time.tv_usec-start_time.tv_usec)/1000000.;

  if (rank == 0)
    printf("Initialization (%s) done in %f s\n",
        this->get_classname(), total_usecs);
}


void FFT3DWithCUFFT::destroy(void)
{
  // cout << "Object is being destroyed" << endl;
cudaFree(data);
cudaFree(datar);
cufftDestroy(plan);
cufftDestroy(plan1);
}


FFT3DWithCUFFT::~FFT3DWithCUFFT(void)
{
}


char const* FFT3DWithCUFFT::get_classname()
{ return "FFT3DWithCUFFT";}


myreal FFT3DWithCUFFT::compute_energy_from_X(myreal* fieldX)
{
  int ii,jj,kk;
  myreal energy = 0.;
  myreal energy1, energy2;

  for (ii=0; ii<nX0; ii++)
    {
    energy1=0.;
    for (jj=0; jj<nX1; jj++)
      {
      energy2=0.;
      for (kk=0; kk<nX2; kk++)      
        energy2 += pow(fieldX[(ii*nX1+jj)*nX2+kk], 2);
      energy1 += energy2/nX2;
      }
    energy += energy1 / nX1;
    }
  //cout << "energyX=" << energy / nX0 / 2 << endl;

  return (myreal) (energy / nX0 / 2);
}


myreal FFT3DWithCUFFT::compute_energy_from_K(mycomplex* fieldK)
{
  int i0, i1, i2;
  double energy = 0;
  double energy0 = 0;

  // modes i1_seq = iKx = last = nK1 - 1
  i2 = nK2 - 1;
  for (i0=0; i0<nK0; i0++)
    for (i1=0; i1<nK1; i1++)
      energy += (double) square_abs(fieldK[i2 + (i1 + i0*nK1)*nK2]);
      // we must divide by 2 ==> after
  
    energy *= 0.5; //divide by 2!!!

  // other modes
  for (i0=0; i0<nK0; i0++)
    for (i1=0; i1<nK1; i1++)
      for (i2=1; i2<nK2-1; i2++)
        energy += (double) square_abs(fieldK[i2 + (i1 + i0*nK1)*nK2]);
    
  // modes i1_seq = iKx = 0
  i2 = 0;
  for (i0=0; i0<nK0; i0++)
    for (i1=0; i1<nK1; i1++)
      energy0 += (double) square_abs(fieldK[(i1 + i0*nK1)*nK2]);
      // we must divide by 2 ==> after

  energy += energy0/2.;

  //cout << "energyK=" << energy<<  endl;
  return (myreal) energy;
}


myreal FFT3DWithCUFFT::sum_wavenumbers_double(myreal* fieldK)
{
  int i0, i1, i2;
  double sum = 0;
  double sum0 = 0;

  // modes i1_seq = iKx = last = nK1 - 1
  i2 = nK2 - 1;
  for (i0=0; i0<nK0; i0++)
    for (i1=0; i1<nK1; i1++)
      sum += (double) fieldK[i2 + (i1 + i0*nK1)*nK2];
      // we must divide by 2 ==> after
  
    sum *= 0.5; //divide by 2!!!

  // other modes
  for (i0=0; i0<nK0; i0++)
    for (i1=0; i1<nK1; i1++)
      for (i2=1; i2<nK2-1; i2++)
        sum += (double) fieldK[i2 + (i1 + i0*nK1)*nK2];
    
  // modes i1_seq = iKx = 0
  i2 = 0;
  for (i0=0; i0<nK0; i0++)
    for (i1=0; i1<nK1; i1++)
      sum0 += (double) fieldK[(i1 + i0*nK1)*nK2];
      // we must divide by 2 ==> after

  sum += sum0/2.;

  return (myreal) 2.*sum;
}


void FFT3DWithCUFFT::sum_wavenumbers_complex(mycomplex* fieldK, mycomplex* result)
{
  int i0, i1, i2;
  mycomplex sum = 0;
  mycomplex sum0 = 0;

  // modes i1_seq = iKx = last = nK1 - 1
  i2 = nK2 - 1;
  for (i0=0; i0<nK0; i0++)
    for (i1=0; i1<nK1; i1++)
      sum += fieldK[i2 + (i1 + i0*nK1)*nK2];
      // we must divide by 2 ==> after
  
    sum *= 0.5; //divide by 2!!!

  // other modes
  for (i0=0; i0<nK0; i0++)
    for (i1=0; i1<nK1; i1++)
      for (i2=1; i2<nK2-1; i2++)
        sum += fieldK[i2 + (i1 + i0*nK1)*nK2];
    
  // modes i1_seq = iKx = 0
  i2 = 0;
  for (i0=0; i0<nK0; i0++)
    for (i1=0; i1<nK1; i1++)
      sum0 += fieldK[(i1 + i0*nK1)*nK2];
      // we must divide by 2 ==> after

  sum += sum0/2.;

  *result = 2.*sum;
}


myreal FFT3DWithCUFFT::compute_mean_from_K(mycomplex* fieldK)
{
  myreal mean;
  mean = real(fieldK[0]);

  return mean;
}


void FFT3DWithCUFFT::fft(myreal *fieldX, mycomplex *fieldK)
{
  
  
  // cout << "FFT3DWithCUFFT::fft" << endl;
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
  myreal norm = inv_coef_norm;
  //  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
  int threadsPerBlock = 256;
  int blocksPerGrid =(nK0 * nK1 * nK2 + threadsPerBlock - 1) / threadsPerBlock;
  vectorNorm<<<blocksPerGrid, threadsPerBlock>>>(norm, data, nK0 * nK1 * nK2 );
  

  // Copy host device to memory
  checkCudaErrors(cudaMemcpy(fieldK, data, mem_size, cudaMemcpyDeviceToHost));


}


void FFT3DWithCUFFT::ifft(mycomplex *fieldK, myreal *fieldX)
{
  //cout << "FFT3DWithCUFFT::ifft" << endl;
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

void FFT3DWithCUFFT::ifft_destroy(mycomplex *fieldK, myreal *fieldX)
{
  //cout << "FFT3DWithCUFFT::ifft" << endl;
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


