
#include <fft3d_with_fftw3d.h>

FFT3DWithFFTW3D::FFT3DWithFFTW3D(int argN0, int argN1, int argN2)
    : BaseFFT3D::BaseFFT3D(argN0, argN1, argN2) {
  chrono::duration<double> clocktime_in_sec;

  this->_init();
#ifdef OMP
#ifdef SINGLE_PREC
  fftwf_init_threads();
#else
  fftw_init_threads();
#endif
#endif
  /* y corresponds to dim 0 in physical space */
  /* y corresponds to dim 1 in physical space */
  /* x corresponds to dim 2 in physical space */
  nz = N0;
  ny = N1;
  nx = N2;

  nX0 = N0;
  nX1 = N1;
  nX2 = N2;
  nX0loc = nX0;
  nX1loc = nX1;
  nX2loc = nX2;

  nKx = nx / 2 + 1;
  nKy = ny;
  nKz = nz;

  /* This 3D fft is NOT transposed */
  nK0 = nKz;
  nK0loc = nK0;
  nK1 = nKy;
  nK1loc = nK1;
  nK2 = nKx;
  nK2loc = nK2;

  flags = FFTW_MEASURE;
  /*    flags = FFTW_ESTIMATE;*/
  /*    flags = FFTW_PATIENT;*/

#ifdef SINGLE_PREC
  arrayX = fftwf_alloc_real(nX0 * nX1 * nX2);
  arrayK = reinterpret_cast<mycomplex *>(fftwf_alloc_complex(nK0 * nK1 * nK2));
#else
  arrayX = (myreal *)fftw_malloc(sizeof(myreal) * nX0 * nX1 * nX2);
  arrayK = reinterpret_cast<mycomplex *>(fftw_malloc(sizeof(mycomplex) * nK0 * nK1 * nK2));
#endif

  auto start_time = chrono::high_resolution_clock::now();
#ifdef OMP
#ifdef SINGLE_PREC
  fftwf_plan_with_nthreads(omp_get_max_threads());
#else
  fftw_plan_with_nthreads(omp_get_max_threads());
#endif
#endif

#ifdef SINGLE_PREC
  plan_r2c = fftwf_plan_dft_r2c_3d(
      N0, N1, N2, arrayX, reinterpret_cast<mycomplex_fftw *>(arrayK), flags);

  plan_c2r = fftwf_plan_dft_c2r_3d(
      N0, N1, N2, reinterpret_cast<mycomplex_fftw *>(arrayK), arrayX, flags);
#else
  plan_r2c = fftw_plan_dft_r2c_3d(
      N0, N1, N2, arrayX, reinterpret_cast<mycomplex_fftw *>(arrayK), flags);

  plan_c2r = fftw_plan_dft_c2r_3d(
      N0, N1, N2, reinterpret_cast<mycomplex_fftw *>(arrayK), arrayX, flags);
#endif

  auto end_time = chrono::high_resolution_clock::now();

  clocktime_in_sec = end_time - start_time;

  if (rank == 0)
    printf("Initialization (%s) done in %f s\n", this->get_classname(),
           clocktime_in_sec.count());
}

void FFT3DWithFFTW3D::destroy(void) {
  // cout << "Object is being destroyed" << endl;
#ifdef SINGLE_PREC
  fftwf_destroy_plan(plan_r2c);
  fftwf_destroy_plan(plan_c2r);
  fftwf_free(arrayX);
  fftwf_free(arrayK);
#ifdef OMP
  fftwf_cleanup_threads();
#endif
#else
  fftw_destroy_plan(plan_r2c);
  fftw_destroy_plan(plan_c2r);
  fftw_free(arrayX);
  fftw_free(arrayK);
#ifdef OMP
  fftw_cleanup_threads();
#endif
#endif
}

FFT3DWithFFTW3D::~FFT3DWithFFTW3D(void) {}

char const *FFT3DWithFFTW3D::get_classname() { return "FFT3DWithFFTW3D"; }

myreal FFT3DWithFFTW3D::compute_energy_from_X(myreal *fieldX) {
  int ii, jj, kk;
  myreal energy = 0.;
  myreal energy1, energy2;

  for (ii = 0; ii < nX0; ii++) {
    energy1 = 0.;
    for (jj = 0; jj < nX1; jj++) {
      energy2 = 0.;
      for (kk = 0; kk < nX2; kk++)
        energy2 += pow(fieldX[(ii * nX1 + jj) * nX2 + kk], 2);
      energy1 += energy2 / nX2;
    }
    energy += energy1 / nX1;
  }
  // cout << "energyX=" << energy / nX0 / 2 << endl;

  return energy / nX0 / 2;
}

myreal FFT3DWithFFTW3D::compute_energy_from_K(mycomplex *fieldK) {
  int i0, i1, i2;
  double energy = 0;
  double energy_tmp = 0;

  // modes i2 = iKx = 0
  i2 = 0;
  for (i0 = 0; i0 < nK0; i0++)
    for (i1 = 0; i1 < nK1; i1++)
      energy_tmp += (double)square_abs(fieldK[(i1 + i0 * nK1) * nK2]);

  energy = energy_tmp / 2;

  // modes i2 = iKx = last = nK2 - 1
  i2 = nK2 - 1;
  energy_tmp = 0.;
  for (i0 = 0; i0 < nK0; i0++)
    for (i1 = 0; i1 < nK1; i1++)
      energy_tmp += (double)square_abs(fieldK[i2 + (i1 + i0 * nK1) * nK2]);

  if (N2 % 2 == 0)
    energy += energy_tmp / 2;
  else
    energy += energy_tmp;

  // other modes
  for (i0 = 0; i0 < nK0; i0++)
    for (i1 = 0; i1 < nK1; i1++)
      for (i2 = 1; i2 < nK2 - 1; i2++)
        energy += (double)square_abs(fieldK[i2 + (i1 + i0 * nK1) * nK2]);

  return (myreal)energy;
}

myreal FFT3DWithFFTW3D::sum_wavenumbers_double(myreal *fieldK) {
  int i0, i1, i2;
  myreal sum = 0;
  myreal sum_tmp = 0;

  // modes i2 = iKx = 0
  i2 = 0;
  for (i0 = 0; i0 < nK0; i0++)
    for (i1 = 0; i1 < nK1; i1++)
      sum_tmp += fieldK[(i1 + i0 * nK1) * nK2];

  sum = sum_tmp / 2;

  // modes i2 = iKx = last = nK2 - 1
  i2 = nK2 - 1;
  sum_tmp = 0.;
  for (i0 = 0; i0 < nK0; i0++)
    for (i1 = 0; i1 < nK1; i1++)
      sum_tmp += fieldK[i2 + (i1 + i0 * nK1) * nK2];

  if (N2 % 2 == 0)
    sum += sum_tmp / 2;
  else
    sum += sum_tmp;

  // other modes
  for (i0 = 0; i0 < nK0; i0++)
    for (i1 = 0; i1 < nK1; i1++)
      for (i2 = 1; i2 < nK2 - 1; i2++)
        sum += fieldK[i2 + (i1 + i0 * nK1) * nK2];

  return sum * 2.;
}

void FFT3DWithFFTW3D::sum_wavenumbers_complex(mycomplex *fieldK,
                                              mycomplex *result) {
  int i0, i1, i2;
  mycomplex sum = 0;
  mycomplex sum_tmp = 0;
  // modes i2 = iKx = 0
  i2 = 0;
  for (i0 = 0; i0 < nK0; i0++)
    for (i1 = 0; i1 < nK1; i1++)
      sum_tmp += fieldK[(i1 + i0 * nK1) * nK2];

  sum = sum_tmp;
  sum *= 0.5;

  // modes i2 = iKx = last = nK2 - 1
  i2 = nK2 - 1;
  sum_tmp = 0.;
  for (i0 = 0; i0 < nK0; i0++)
    for (i1 = 0; i1 < nK1; i1++)
      sum_tmp += fieldK[i2 + (i1 + i0 * nK1) * nK2];

  if (N2 % 2 == 0)
    sum_tmp /= 2.0;
  sum += sum_tmp;

  // other modes
  for (i0 = 0; i0 < nK0; i0++)
    for (i1 = 0; i1 < nK1; i1++)
      for (i2 = 1; i2 < nK2 - 1; i2++)
        sum += fieldK[i2 + (i1 + i0 * nK1) * nK2];
  sum *= 2.0;
  *result = sum;
}

myreal FFT3DWithFFTW3D::compute_mean_from_K(mycomplex *fieldK) {
  myreal mean = real(fieldK[0]);
  return mean;
}

void FFT3DWithFFTW3D::fft(myreal *fieldX, mycomplex *fieldK) {
  int ii;
  // cout << "FFT3DWithFFTW3D::fft" << endl;

#ifdef SINGLE_PREC
  fftwf_execute_dft_r2c(plan_r2c, fieldX,
                        reinterpret_cast<mycomplex_fftw *>(arrayK));
#else
  fftw_execute_dft_r2c(plan_r2c, fieldX,
                       reinterpret_cast<mycomplex_fftw *>(arrayK));
#endif

  for (ii = 0; ii < nK0 * nK1 * nK2; ii++)
    fieldK[ii] = arrayK[ii] * inv_coef_norm;
}

void FFT3DWithFFTW3D::ifft(mycomplex *fieldK, myreal *fieldX) {
  // cout << "FFT3DWithFFTW3D::ifft" << endl;
  memcpy(arrayK, fieldK, nK0 * nK1 * nK2 * sizeof(mycomplex));
#ifdef SINGLE_PREC
  fftwf_execute_dft_c2r(plan_c2r, reinterpret_cast<mycomplex_fftw *>(arrayK),
                        fieldX);
#else
  fftw_execute_dft_c2r(plan_c2r, reinterpret_cast<mycomplex_fftw *>(arrayK),
                       fieldX);
#endif
}

void FFT3DWithFFTW3D::ifft_destroy(mycomplex *fieldK, myreal *fieldX) {
#ifdef SINGLE_PREC
  fftwf_execute_dft_c2r(plan_c2r, reinterpret_cast<mycomplex_fftw *>(fieldK),
                        fieldX);
#else
  fftw_execute_dft_c2r(plan_c2r, reinterpret_cast<mycomplex_fftw *>(fieldK),
                       fieldX);
#endif
}
