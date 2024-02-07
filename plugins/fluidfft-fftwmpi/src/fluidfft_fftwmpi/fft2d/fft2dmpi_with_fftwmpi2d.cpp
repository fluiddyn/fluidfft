
#include <fft2dmpi_with_fftwmpi2d.h>

FFT2DMPIWithFFTWMPI2D::FFT2DMPIWithFFTWMPI2D(int argN0, int argN1)
    : BaseFFT2DMPI::BaseFFT2DMPI(argN0, argN1) {
  int nK0_loc_iszero = 0;
  int count_nK0loc_zero = 0;
  chrono::duration<double> clocktime_in_sec;
  ptrdiff_t local_nX0;
  ptrdiff_t local_nK0;

  this->_init();
#ifdef SINGLE_PREC
  fftwf_mpi_init();

  /* get local data size and allocate */
  alloc_local = fftwf_mpi_local_size_2d_transposed(
      N0, N1 / 2 + 1, MPI_COMM_WORLD, &local_nX0, &local_X0_start, &local_nK0,
      &local_K0_start);
#else
  fftw_mpi_init();

  /* get local data size and allocate */
  alloc_local = fftw_mpi_local_size_2d_transposed(
      N0, N1 / 2 + 1, MPI_COMM_WORLD, &local_nX0, &local_X0_start, &local_nK0,
      &local_K0_start);
#endif
  /* y corresponds to dim 0 in physical space */
  /* x corresponds to dim 1 in physical space */
  ny = N0;
  nx = N1;

  nX0 = N0;
  nX1 = N1;
  nX0loc = local_nX0;
  nXyloc = nX0loc;
  nX1_pad = 2 * (N1 / 2 + 1);

  nKx = nx / 2 + 1;
  nKxloc = local_nK0;
  nKy = ny;

  /* This 2D fft is transposed */
  is_transposed = 1;
  nK0 = nKx;
  nK0loc = nKxloc;
  nK1 = nKy;

  size_fieldK = nKxloc * nKy;

  /* Case nK0loc ==0 */
  if (nK0loc == 0) {
    nK0_loc_iszero = 1;
  }
  MPI_Allreduce(&nK0_loc_iszero, &count_nK0loc_zero, 1, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);
  last_rank_nozero = nb_proc - 1 - count_nK0loc_zero;

  flags = FFTW_MEASURE;
  /*    flags = FFTW_ESTIMATE;*/
  /*    flags = FFTW_PATIENT;*/

#ifdef SINGLE_PREC
  arrayX = fftwf_alloc_real(2 * alloc_local);
  arrayK = fftwf_alloc_complex(alloc_local);

  auto start_time = chrono::high_resolution_clock::now();

  plan_r2c = fftwf_mpi_plan_dft_r2c_2d(N0, N1, arrayX, arrayK, MPI_COMM_WORLD,
                                       flags | FFTW_MPI_TRANSPOSED_OUT);

  plan_c2r = fftwf_mpi_plan_dft_c2r_2d(N0, N1, arrayK, arrayX, MPI_COMM_WORLD,
                                       flags | FFTW_MPI_TRANSPOSED_IN);
#else
  arrayX = (myreal *)fftw_malloc(sizeof(myreal) * 2 * alloc_local);
  arrayK = reinterpret_cast<mycomplex *>(fftw_malloc(sizeof(mycomplex) * alloc_local));

  auto start_time = chrono::high_resolution_clock::now();

  plan_r2c = fftw_mpi_plan_dft_r2c_2d(
      N0, N1, arrayX, reinterpret_cast<mycomplex_fftw *>(arrayK),
      MPI_COMM_WORLD, flags | FFTW_MPI_TRANSPOSED_OUT);

  plan_c2r = fftw_mpi_plan_dft_c2r_2d(
      N0, N1, reinterpret_cast<mycomplex_fftw *>(arrayK), arrayX,
      MPI_COMM_WORLD, flags | FFTW_MPI_TRANSPOSED_IN);
#endif
  auto end_time = chrono::high_resolution_clock::now();

  clocktime_in_sec = end_time - start_time;

  if (rank == 0)
    printf("Initialization (%s) done in %f s\n", this->get_classname(),
           clocktime_in_sec.count());
}

void FFT2DMPIWithFFTWMPI2D::destroy(void) {
  // cout << "Object is being destroyed" << endl;
#ifdef SINGLE_PREC
  fftwf_destroy_plan(plan_r2c);
  fftwf_destroy_plan(plan_c2r);
  fftwf_free(arrayX);
  fftwf_free(arrayK);
#else
  fftw_destroy_plan(plan_r2c);
  fftw_destroy_plan(plan_c2r);
  fftw_free(arrayX);
  fftw_free(arrayK);
#endif
}

FFT2DMPIWithFFTWMPI2D::~FFT2DMPIWithFFTWMPI2D(void) {}

bool FFT2DMPIWithFFTWMPI2D::are_parameters_bad() {
  if (N0 % nb_proc != 0) {
    if (rank == 0)
      cout << "bad parameters: (N0 % nb_proc != 0); (N0=" << N0
           << ", nb_proc=" << nb_proc << ")" << endl;
    return 1;
  }
  if (N1 / 2 % nb_proc != 0) {
    if (rank == 0)
      cout << "bad parameters: (N1/2 % nb_proc != 0); (N1/2=" << N1 / 2
           << ", nb_proc=" << nb_proc << ")" << endl;
    return 1;
  }
  return 0;
}

char const *FFT2DMPIWithFFTWMPI2D::get_classname() {
  return "FFT2DMPIWithFFTWMPI2D";
}

myreal FFT2DMPIWithFFTWMPI2D::compute_energy_from_X(myreal *fieldX) {
  int i0, i1;
  myreal energy_loc = 0;
  myreal energy;

  for (i0 = 0; i0 < nX0loc; i0++)
    for (i1 = 0; i1 < nX1; i1++)
      energy_loc += pow(fieldX[i1 + i0 * nX1], 2);
#ifdef SINGLE_PREC
  MPI_Allreduce(&energy_loc, &energy, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
#else
  MPI_Allreduce(&energy_loc, &energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
  return energy / 2 * inv_coef_norm;
}

myreal FFT2DMPIWithFFTWMPI2D::compute_energy_from_K(mycomplex *fieldK) {
  int i0, i1, i_tmp;
  myreal energy_loc = 0;
  myreal energy_tmp = 0;
  myreal energy;

  if (nK0loc != 0) {
    // modes i0 = iKx = 0
    i0 = 0;
    for (i1 = 0; i1 < nK1; i1++)
      energy_tmp += pow(abs(fieldK[i1]), 2);

    // if iKx == 0 | nK0loc == 1 && iKx=last
    if ((rank == 0) |
        ((rank == last_rank_nozero) & (nK0loc == 1))) // i.e. if iKx == 0
      energy_loc = energy_tmp / 2;
    else
      energy_loc = energy_tmp;

    if (nK0loc > 1) {
      // modes i0 = iKx = last = nK0loc - 1
      i0 = nK0loc - 1;
      energy_tmp = 0.;
      i_tmp = i0 * nK1;
      for (i1 = 0; i1 < nK1; i1++)
        energy_tmp += pow(abs(fieldK[i1 + i_tmp]), 2);
      if (rank == last_rank_nozero)
        energy_loc += energy_tmp / 2;
      else
        energy_loc += energy_tmp;
      // other modes
      for (i0 = 1; i0 < nK0loc - 1; i0++)
        for (i1 = 0; i1 < nK1; i1++)
          energy_loc += pow(abs(fieldK[i1 + i0 * nK1]), 2);
    }
  }
#ifdef SINGLE_PREC
  MPI_Allreduce(&energy_loc, &energy, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
#else
  MPI_Allreduce(&energy_loc, &energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
  return energy;
}

myreal FFT2DMPIWithFFTWMPI2D::sum_wavenumbers(myreal *fieldK) {
  int i0, i1, i_tmp;
  myreal sum_loc = 0;
  myreal sum_tmp = 0;
  myreal sum_tot;

  // modes i0 = iKx = 0
  i0 = 0;
  for (i1 = 0; i1 < nK1; i1++)
    sum_tmp += fieldK[i1];

  // if (local_K0_start == 0)  // i.e. if iKx == 0
  if ((rank == 0) |
      ((rank == last_rank_nozero) & (nK0loc == 1))) // i.e. if iKx == 0
    sum_loc = sum_tmp / 2;
  else
    sum_loc = sum_tmp;

  if (nK0loc > 1) {
    // modes i0 = iKx = last = nK0loc - 1
    i0 = nK0loc - 1;
    sum_tmp = 0.;
    i_tmp = i0 * nK1;
    for (i1 = 0; i1 < nK1; i1++)
      sum_tmp += fieldK[i1 + i_tmp];

    if (rank == last_rank_nozero)
      sum_loc += sum_tmp / 2;
    else
      sum_loc += sum_tmp;

    // other modes
    for (i0 = 1; i0 < nK0loc - 1; i0++)
      for (i1 = 0; i1 < nK1; i1++)
        sum_loc += fieldK[i1 + i0 * nK1];
  }
#ifdef SINGLE_PREC
  MPI_Allreduce(&sum_loc, &sum_tot, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
#else
  MPI_Allreduce(&sum_loc, &sum_tot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
  return 2 * sum_tot;
}

myreal FFT2DMPIWithFFTWMPI2D::compute_mean_from_X(myreal *fieldX) {
  myreal mean, local_mean;
  int ii;
  local_mean = 0.;

  for (ii = 0; ii < nX0loc * nX1; ii++)
    local_mean += fieldX[ii];
#ifdef SINGLE_PREC
  MPI_Allreduce(&local_mean, &mean, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
#else
  MPI_Allreduce(&local_mean, &mean, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
  return mean * inv_coef_norm;
}

myreal FFT2DMPIWithFFTWMPI2D::compute_mean_from_K(mycomplex *fieldK) {
  myreal mean, local_mean;
  if (local_K0_start == 0)
    local_mean = real(fieldK[0]);
  else
    local_mean = 0.;
#ifdef SINGLE_PREC
  MPI_Allreduce(&local_mean, &mean, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
#else
  MPI_Allreduce(&local_mean, &mean, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
  return mean;
}

void FFT2DMPIWithFFTWMPI2D::fft(myreal *fieldX, mycomplex *fieldK) {
  int i0, i1;
  // cout << "FFT2DMPIWithFFTWMPI2D::fft" << endl;

  for (i0 = 0; i0 < nX0loc; i0++)
    for (i1 = 0; i1 < nX1; i1++)
      arrayX[i1 + i0 * nX1_pad] = fieldX[i1 + i0 * nX1];

#ifdef SINGLE_PREC
  fftwf_execute(plan_r2c);
#else
  fftw_execute(plan_r2c);
#endif
  for (i0 = 0; i0 < nK0loc; i0++)
    for (i1 = 0; i1 < nK1; i1++)
      fieldK[i1 + i0 * nK1] = arrayK[i1 + i0 * nK1] * inv_coef_norm;
}

void FFT2DMPIWithFFTWMPI2D::ifft(mycomplex *fieldK, myreal *fieldX) {
  int i0, i1;
  // cout << "FFT2DMPIWithFFTWMPI2D::ifft" << endl;
  memcpy(arrayK, fieldK, size_fieldK * sizeof(mycomplex));
#ifdef SINGLE_PREC
  fftwf_execute(plan_c2r);
#else
  fftw_execute(plan_c2r);
#endif
  for (i0 = 0; i0 < nX0loc; i0++)
    for (i1 = 0; i1 < nX1; i1++)
      fieldX[i1 + i0 * nX1] = arrayX[i1 + i0 * nX1_pad];
}
