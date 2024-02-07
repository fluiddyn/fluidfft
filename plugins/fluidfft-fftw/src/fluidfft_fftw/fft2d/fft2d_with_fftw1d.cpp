

#include <iostream>
#include <chrono>
using namespace std;

#include <stdlib.h>
#include <string.h>

#include <fft2d_with_fftw1d.h>

FFT2DWithFFTW1D::FFT2DWithFFTW1D(int argN0, int argN1)
    : BaseFFT2D::BaseFFT2D(argN0, argN1) {
  chrono::duration<double> clocktime_in_sec;
  // int iX0;
  int istride = 1, ostride = 1;
  int howmany, sign;

  this->_init();

  /* y corresponds to dim 0 in physical space */
  /* x corresponds to dim 1 in physical space */
  ny = N0;
  nx = N1;

  nX0 = N0;
  nX1 = N1;
  nX0loc = nX0;
  nXyloc = nX0loc;

  nKx = nx / 2 + 1;
  nKxloc = nKx;
  nKy = ny;

  /* This 2D fft is transposed */
  is_transposed = 1;
  nK0 = nKx;
  nK0loc = nK0;
  nK1 = nKy;

  flags = FFTW_MEASURE;
/*    flags = FFTW_ESTIMATE;*/
/*    flags = FFTW_PATIENT;*/
#ifdef SINGLE_PREC
  arrayX = (myreal *)fftwf_malloc(sizeof(myreal) * nX0loc * nX1);
  arrayK_pR = (mycomplex *)fftwf_malloc(sizeof(mycomplex) * nX0loc * nKx);
  arrayK_pC = (mycomplex *)fftwf_malloc(sizeof(mycomplex) * nKxloc * N0);

  auto start_time = chrono::high_resolution_clock::now();

  howmany = nX0loc;

  plan_r2c = fftwf_plan_many_dft_r2c(1, &N1, howmany, arrayX, NULL, istride, N1,
                                     arrayK_pR, NULL, ostride, nKx, flags);

  plan_c2r = fftwf_plan_many_dft_c2r(1, &N1, howmany, arrayK_pR, NULL, istride,
                                     nKx, arrayX, NULL, ostride, N1, flags);

  howmany = nKxloc;
  sign = FFTW_FORWARD;
  plan_c2c_fwd =
      fftwf_plan_many_dft(1, &N0, howmany, arrayK_pC, &N0, istride, N0,
                          arrayK_pC, &N0, ostride, N0, sign, flags);

  sign = FFTW_BACKWARD;
  plan_c2c_bwd =
      fftwf_plan_many_dft(1, &N0, howmany, arrayK_pC, &N0, istride, N0,
                          arrayK_pC, &N0, ostride, N0, sign, flags);
#else
  arrayX = (myreal *)fftw_malloc(sizeof(myreal) * nX0loc * nX1);
  arrayK_pR = (mycomplex *)fftw_malloc(sizeof(mycomplex) * nX0loc * nKx);
  arrayK_pC = (mycomplex *)fftw_malloc(sizeof(mycomplex) * nKxloc * N0);

  auto start_time = chrono::high_resolution_clock::now();

  howmany = nX0loc;

  plan_r2c = fftw_plan_many_dft_r2c(
      1, &N1, howmany, arrayX, NULL, istride, N1,
      reinterpret_cast<mycomplex_fftw *>(arrayK_pR), NULL, ostride, nKx, flags);

  plan_c2r = fftw_plan_many_dft_c2r(
      1, &N1, howmany, reinterpret_cast<mycomplex_fftw *>(arrayK_pR), NULL,
      istride, nKx, arrayX, NULL, ostride, N1, flags);

  howmany = nKxloc;
  sign = FFTW_FORWARD;
  plan_c2c_fwd = fftw_plan_many_dft(
      1, &N0, howmany, reinterpret_cast<mycomplex_fftw *>(arrayK_pC), &N0,
      istride, N0, reinterpret_cast<mycomplex_fftw *>(arrayK_pC), &N0, ostride,
      N0, sign, flags);

  sign = FFTW_BACKWARD;
  plan_c2c_bwd = fftw_plan_many_dft(
      1, &N0, howmany, reinterpret_cast<mycomplex_fftw *>(arrayK_pC), &N0,
      istride, N0, reinterpret_cast<mycomplex_fftw *>(arrayK_pC), &N0, ostride,
      N0, sign, flags);
#endif
  auto end_time = chrono::high_resolution_clock::now();

  clocktime_in_sec = end_time - start_time;

  if (rank == 0)
    printf("Initialization (%s) done in %f s\n", this->get_classname(),
           clocktime_in_sec.count());
}

void FFT2DWithFFTW1D::destroy(void) {
  // cout << "Object is being destroyed" << endl;
#ifdef SINGLE_PREC
  fftwf_destroy_plan(plan_r2c);
  fftwf_destroy_plan(plan_c2c_fwd);
  fftwf_destroy_plan(plan_c2c_bwd);
  fftwf_destroy_plan(plan_c2r);
  fftwf_free(arrayX);
  fftwf_free(arrayK_pR);
  fftwf_free(arrayK_pC);
#else
  fftw_destroy_plan(plan_r2c);
  fftw_destroy_plan(plan_c2c_fwd);
  fftw_destroy_plan(plan_c2c_bwd);
  fftw_destroy_plan(plan_c2r);
  fftw_free(arrayX);
  fftw_free(arrayK_pR);
  fftw_free(arrayK_pC);
#endif
}

FFT2DWithFFTW1D::~FFT2DWithFFTW1D(void) {
  // cout << "Object is being deleted" << endl;
}

char const *FFT2DWithFFTW1D::get_classname() { return "FFT2DWithFFTW1D"; }

myreal FFT2DWithFFTW1D::compute_energy_from_X(myreal *fieldX) {
  int ii;
  myreal energy = 0;

  for (ii = 0; ii < nX0loc * nX1; ii++)
    energy += pow(fieldX[ii], 2);

  return energy / 2 * inv_coef_norm;
}

myreal FFT2DWithFFTW1D::compute_energy_from_K(mycomplex *fieldK) {
  int i0, i1;
  myreal energy = 0;
  myreal energy_tmp = 0;

  // modes i0 = iKx = 0
  i0 = 0;
  for (i1 = 0; i1 < nK1; i1++)
    energy_tmp += pow(abs(fieldK[i1]), 2);

  energy = energy_tmp / 2;

  // modes i0 = iKx = last = nK0-1
  i0 = nK0loc - 1;
  energy_tmp = 0;
  for (i1 = 0; i1 < nK1; i1++)
    energy_tmp += pow(abs(fieldK[i1 + i0 * nK1]), 2);

  if (N1 % 2 == 0)
    energy += energy_tmp / 2;
  else
    energy += energy_tmp;

  // other modes
  for (i0 = 1; i0 < nK0loc - 1; i0++)
    for (i1 = 0; i1 < nK1; i1++)
      energy += pow(abs(fieldK[i1 + i0 * nK1]), 2);

  return energy;
}

myreal FFT2DWithFFTW1D::sum_wavenumbers(myreal *fieldK) {
  int i0, i1;
  myreal sum = 0;
  myreal sum_tmp = 0;

  // modes i0 = iKx = 0
  i0 = 0;
  for (i1 = 0; i1 < nK1; i1++)
    sum_tmp += fieldK[i1];

  sum = sum_tmp / 2;

  // modes i0 = iKx = last = nK0-1
  i0 = nK0loc - 1;
  sum_tmp = 0;
  for (i1 = 0; i1 < nK1; i1++)
    sum_tmp += fieldK[i1 + i0 * nK1];

  if (N1 % 2 == 0)
    sum += sum_tmp / 2;
  else
    sum += sum_tmp;

  // other modes

  for (i0 = 1; i0 < nK0loc - 1; i0++)
    for (i1 = 0; i1 < nK1; i1++)
      sum += fieldK[i1 + i0 * nK1];

  return 2 * sum;
}

myreal FFT2DWithFFTW1D::compute_mean_from_X(myreal *fieldX) {
  myreal mean = 0.;
  int ii;

  for (ii = 0; ii < nX0loc * nX1; ii++)
    mean += fieldX[ii];

  return mean * inv_coef_norm;
}
myreal FFT2DWithFFTW1D::compute_mean_from_K(mycomplex *fieldK) {
  myreal mean = fieldK[0].real();
  return mean;
}
void FFT2DWithFFTW1D::fft(myreal *fieldX, mycomplex *fieldK) {
  int ii, jj;
  // cout << "FFT2DWithFFTW1D::fft" << endl;
  /*use memcpy(void * destination, void * source, size_t bytes); */
  memcpy(arrayX, fieldX, nX0loc * nX1 * sizeof(myreal));
#ifdef SINGLE_PREC
  fftwf_execute(plan_r2c);
#else
  fftw_execute(plan_r2c);
#endif
  /*    second step: transpose...*/
  for (ii = 0; ii < nX0; ++ii)
    for (jj = 0; jj < nKx; ++jj)
      arrayK_pC[jj * nX0 + ii] = arrayK_pR[ii * nKx + jj];
#ifdef SINGLE_PREC
  fftwf_execute(plan_c2c_fwd);
#else
  fftw_execute(plan_c2c_fwd);
#endif

  for (ii = 0; ii < nKxloc * nKy; ii++)
    fieldK[ii] = arrayK_pC[ii] * inv_coef_norm;
}
void FFT2DWithFFTW1D::ifft(mycomplex *fieldK, myreal *fieldX) {
  int ii, jj;
  // cout << "FFT2DWithFFTW1D::ifft" << endl;
  memcpy(arrayK_pC, fieldK, nKxloc * nKy * sizeof(mycomplex));
#ifdef SINGLE_PREC
  fftwf_execute(plan_c2c_bwd);
#else
  fftw_execute(plan_c2c_bwd);
#endif
  /* second step: transpose...*/
  for (ii = 0; ii < nKx; ++ii)
    for (jj = 0; jj < nX0; ++jj)
      arrayK_pR[jj * nKx + ii] = arrayK_pC[ii * nX0 + jj];

#ifdef SINGLE_PREC
  fftwf_execute(plan_c2r);
#else
  fftw_execute(plan_c2r);
#endif
  memcpy(fieldX, arrayX, nX0loc * nX1 * sizeof(myreal));
}
