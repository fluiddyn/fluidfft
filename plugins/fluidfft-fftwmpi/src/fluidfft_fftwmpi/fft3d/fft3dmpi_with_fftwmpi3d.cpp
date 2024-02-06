

#include <iostream>
#include <chrono>
using namespace std;

#include <stdlib.h>

#ifdef OMP
#include <omp.h>
#endif

#include <fft3dmpi_with_fftwmpi3d.h>

FFT3DMPIWithFFTWMPI3D::FFT3DMPIWithFFTWMPI3D(int argN0, int argN1, int argN2)
    : BaseFFT3DMPI::BaseFFT3DMPI(argN0, argN1, argN2) {
  chrono::duration<double> clocktime_in_sec;
  ptrdiff_t local_nX0; //, local_X0_start;
  ptrdiff_t local_nK0;

  this->_init();
#ifdef SINGLE_PREC
#ifdef OMP
  fftwf_init_threads();
#endif
  fftwf_mpi_init();
  /* get local data size and allocate */
  alloc_local = fftwf_mpi_local_size_3d_transposed(
      N0, N1, N2 / 2 + 1, MPI_COMM_WORLD, &local_nX0, &local_X0_start,
      &local_nK0, &local_K0_start);
#else
#ifdef OMP
  fftw_init_threads();
#endif
  fftw_mpi_init();
  /* get local data size and allocate */
  alloc_local = fftw_mpi_local_size_3d_transposed(
      N0, N1, N2 / 2 + 1, MPI_COMM_WORLD, &local_nX0, &local_X0_start,
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
  nX2loc = nX2;

  nX2_pad = 2 * (N2 / 2 + 1);

  /* This 3D fft is transposed */
  /* in Fourier space: */
  /* ky corresponds to dim 0 */
  /* kz corresponds to dim 1 */
  /* kx corresponds to dim 2 */
  nKx = nx / 2 + 1;
  nK2 = nKx;

  nKy = ny;
  nK0 = nKy;
  nK0loc = local_nK0;

  nK1 = N0;
  nK1loc = nK1;
  nK2loc = nK2;

  size_fieldK = nK0loc * nK1loc * nK2loc;

  flags = FFTW_MEASURE;
  /*    flags = FFTW_ESTIMATE;*/
  /*    flags = FFTW_PATIENT;*/

#ifdef SINGLE_PREC
  arrayX = fftwf_alloc_real(2 * alloc_local);
  arrayK = reinterpret_cast<mycomplex *>(fftwf_alloc_complex(alloc_local));

  auto start_time = chrono::high_resolution_clock::now();
#ifdef OMP
  fftwf_plan_with_nthreads(omp_get_max_threads());
#endif
  plan_r2c =
      fftwf_mpi_plan_dft_r2c_3d(N0, N1, N2, arrayX, reinterpret_cast<mycomplex_fftw *>(arrayK), MPI_COMM_WORLD,
                                flags | FFTW_MPI_TRANSPOSED_OUT);

  plan_c2r =
      fftwf_mpi_plan_dft_c2r_3d(N0, N1, N2, reinterpret_cast<mycomplex_fftw *>(arrayK), arrayX, MPI_COMM_WORLD,
                                flags | FFTW_MPI_TRANSPOSED_IN);
#else
  arrayX = (myreal *)fftw_malloc(sizeof(myreal) * 2 * alloc_local);
  arrayK = reinterpret_cast<mycomplex *>(fftw_malloc(sizeof(mycomplex) * alloc_local));

  auto start_time = chrono::high_resolution_clock::now();
#ifdef OMP
  fftw_plan_with_nthreads(omp_get_max_threads());
#endif
  plan_r2c = fftw_mpi_plan_dft_r2c_3d(
      N0, N1, N2, arrayX, reinterpret_cast<mycomplex_fftw *>(arrayK),
      MPI_COMM_WORLD, flags | FFTW_MPI_TRANSPOSED_OUT);

  plan_c2r = fftw_mpi_plan_dft_c2r_3d(
      N0, N1, N2, reinterpret_cast<mycomplex_fftw *>(arrayK), arrayX,
      MPI_COMM_WORLD, flags | FFTW_MPI_TRANSPOSED_IN);
#endif
  auto end_time = chrono::high_resolution_clock::now();

  clocktime_in_sec = end_time - start_time;

  if (rank == 0)
    printf("Initialization (%s) done in %f s\n", this->get_classname(),
           clocktime_in_sec.count());
}

void FFT3DMPIWithFFTWMPI3D::destroy(void) {
  // cout << "Object is being destroyed" << endl;
#ifdef SINGLE_PREC
  fftwf_destroy_plan(plan_r2c);
  fftwf_destroy_plan(plan_c2r);
  fftwf_free(arrayX);
  fftwf_free(arrayK);
  // fftwf_mpi_cleanup();
#ifdef OMP
  fftwf_cleanup_threads();
#endif
#else
  fftw_destroy_plan(plan_r2c);
  fftw_destroy_plan(plan_c2r);
  fftw_free(arrayX);
  fftw_free(arrayK);
  // fftw_mpi_cleanup();
#ifdef OMP
  fftw_cleanup_threads();
#endif
#endif
}

char const *FFT3DMPIWithFFTWMPI3D::get_classname() {
  return "FFT3DMPIWithFFTWMPI3D";
}

myreal FFT3DMPIWithFFTWMPI3D::compute_energy_from_K(mycomplex *fieldK) {
  int i0, i1, i2;
  double energy_tmp = 0;
  double energy_loc;
  double energy;

  // modes i2 = iKx = 0
  i2 = 0;
  for (i0 = 0; i0 < nK0loc; i0++)
    for (i1 = 0; i1 < nK1; i1++)
      energy_tmp += (double)square_abs(fieldK[(i1 + i0 * nK1) * nK2]);

  energy_loc = energy_tmp / 2.;

  // modes i2 = iKx = last = nK2 - 1
  i2 = nK2 - 1;
  energy_tmp = 0;
  for (i0 = 0; i0 < nK0loc; i0++)
    for (i1 = 0; i1 < nK1; i1++)
      energy_tmp += (double)square_abs(fieldK[i2 + (i1 + i0 * nK1) * nK2]);

  if (N2 % 2 == 0)
    energy_loc += energy_tmp / 2.;
  else
    energy_loc += energy_tmp;

  // other modes
  for (i0 = 0; i0 < nK0loc; i0++)
    for (i1 = 0; i1 < nK1; i1++)
      for (i2 = 1; i2 < nK2 - 1; i2++)
        energy_loc += (double)square_abs(fieldK[i2 + (i1 + i0 * nK1) * nK2]);

  MPI_Allreduce(&energy_loc, &energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  return (myreal)energy;
}

myreal FFT3DMPIWithFFTWMPI3D::sum_wavenumbers_double(myreal *fieldK) {
  int i0, i1, i2;
  double energy_tmp = 0;
  double energy_loc, energy;

  // modes i2 = iKx = 0
  i2 = 0;
  for (i0 = 0; i0 < nK0loc; i0++)
    for (i1 = 0; i1 < nK1; i1++)
      energy_tmp += (double)fieldK[(i1 + i0 * nK1) * nK2];

  energy_loc = energy_tmp / 2.;

  // modes i2 = iKx = last = nK2 - 1
  i2 = nK2 - 1;
  energy_tmp = 0;
  for (i0 = 0; i0 < nK0loc; i0++)
    for (i1 = 0; i1 < nK1; i1++)
      energy_tmp += (double)fieldK[i2 + (i1 + i0 * nK1) * nK2];

  if (N2 % 2 == 0)
    energy_loc += energy_tmp / 2.;
  else
    energy_loc += energy_tmp;

  // other modes
  for (i0 = 0; i0 < nK0loc; i0++)
    for (i1 = 0; i1 < nK1; i1++)
      for (i2 = 1; i2 < nK2 - 1; i2++)
        energy_loc += (double)fieldK[i2 + (i1 + i0 * nK1) * nK2];

  MPI_Allreduce(&energy_loc, &energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  return (myreal)energy * 2.;
}

void FFT3DMPIWithFFTWMPI3D::sum_wavenumbers_complex(mycomplex *fieldK,
                                                    mycomplex *result) {
  int i0, i1, i2;
  mycomplex energy_tmp = 0;
  mycomplex energy_loc, energy;
  // modes i2 = iKx = 0
  i2 = 0;
  for (i0 = 0; i0 < nK0loc; i0++)
    for (i1 = 0; i1 < nK1; i1++)
      energy_tmp += fieldK[(i1 + i0 * nK1) * nK2];
  energy_tmp /= 2.;
  energy_loc = energy_tmp;

  // modes i2 = iKx = last = nK2 - 1
  i2 = nK2 - 1;
  energy_tmp = 0;
  for (i0 = 0; i0 < nK0loc; i0++)
    for (i1 = 0; i1 < nK1; i1++)
      energy_tmp += fieldK[i2 + (i1 + i0 * nK1) * nK2];

  if (N2 % 2 == 0)
    energy_tmp /= 2.;
  energy_loc += energy_tmp;

  // other modes
  for (i0 = 0; i0 < nK0loc; i0++)
    for (i1 = 0; i1 < nK1; i1++)
      for (i2 = 1; i2 < nK2 - 1; i2++)
        energy_loc += fieldK[i2 + (i1 + i0 * nK1) * nK2];

  MPI_Allreduce(&energy_loc, &energy, 1, MPI_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
  // ERREUR PTET LA... COMPLEX OU DOUBLE COMPLEX???
  energy *= 2.;
  *result = energy;
}

void FFT3DMPIWithFFTWMPI3D::fft(myreal *fieldX, mycomplex *fieldK) {
  int i0, i1, i2;
  // cout << "FFT3DMPIWithFFTWMPI3D::fft" << endl;

  for (i0 = 0; i0 < nX0loc; i0++)
    for (i1 = 0; i1 < nX1; i1++)
      for (i2 = 0; i2 < nX2; i2++)
        arrayX[i2 + (i1 + i0 * nX1) * nX2_pad] =
            fieldX[i2 + (i1 + i0 * nX1) * nX2];

#ifdef SINGLE_PREC
  fftwf_execute(plan_r2c);
#else
  fftw_execute(plan_r2c);
#endif

  for (i0 = 0; i0 < nK0loc; i0++)
    for (i1 = 0; i1 < nK1; i1++)
      for (i2 = 0; i2 < nK2; i2++)
        fieldK[i2 + (i1 + i0 * nK1) * nK2] =
            arrayK[i2 + (i1 + i0 * nK1) * nK2] * inv_coef_norm;
}

void FFT3DMPIWithFFTWMPI3D::ifft(mycomplex *fieldK, myreal *fieldX) {
  int i0, i1, i2;
  // cout << "FFT3DMPIWithFFTWMPI3D::ifft" << endl;
  memcpy(arrayK, fieldK, size_fieldK * sizeof(mycomplex));
#ifdef SINGLE_PREC
  fftwf_execute(plan_c2r);
#else
  fftw_execute(plan_c2r);
#endif

  for (i0 = 0; i0 < nX0loc; i0++)
    for (i1 = 0; i1 < nX1; i1++)
      for (i2 = 0; i2 < nX2; i2++)
        fieldX[i2 + (i1 + i0 * nX1) * nX2] =
            arrayX[i2 + (i1 + i0 * nX1) * nX2_pad];
}

void FFT3DMPIWithFFTWMPI3D::ifft_destroy(mycomplex *fieldK, myreal *fieldX) {
  int i0, i1, i2;
  // todo: we are allowed to destroy the input here! No copy!
  memcpy(arrayK, fieldK, size_fieldK * sizeof(mycomplex));
#ifdef SINGLE_PREC
  fftwf_execute(plan_c2r);
#else
  fftw_execute(plan_c2r);
#endif

  for (i0 = 0; i0 < nX0loc; i0++)
    for (i1 = 0; i1 < nX1; i1++)
      for (i2 = 0; i2 < nX2; i2++)
        fieldX[i2 + (i1 + i0 * nX1) * nX2] =
            arrayX[i2 + (i1 + i0 * nX1) * nX2_pad];
}

void FFT3DMPIWithFFTWMPI3D::get_dimX_K(int *d0, int *d1, int *d2) {
  *d0 = 1;
  *d1 = 0;
  *d2 = 2;
}

void FFT3DMPIWithFFTWMPI3D::get_seq_indices_first_K(int *i0, int *i1, int *i2) {
  *i0 = local_K0_start;
  *i1 = 0;
  *i2 = 0;
}

void FFT3DMPIWithFFTWMPI3D::get_seq_indices_first_X(int *i0, int *i1, int *i2) {
  *i0 = local_X0_start;
  *i1 = 0;
  *i2 = 0;
}

bool FFT3DMPIWithFFTWMPI3D::are_parameters_bad() {
  if (N0 / nb_proc == 0) {
    if (rank == 0)
      cout << "bad parameters N0 : nb_proc>N0 (not supported by fftwmpi)"
           << endl;
    return 1;
  }
  return 0;
}
