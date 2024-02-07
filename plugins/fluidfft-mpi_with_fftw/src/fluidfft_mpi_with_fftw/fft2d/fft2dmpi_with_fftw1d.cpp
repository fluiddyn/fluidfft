

#include <fft2dmpi_with_fftw1d.h>

FFT2DMPIWithFFTW1D::FFT2DMPIWithFFTW1D(int argN0, int argN1)
    : BaseFFT2DMPI::BaseFFT2DMPI(argN0, argN1) {
  chrono::duration<double> clocktime_in_sec;
  int iX0;
  int istride = 1, ostride = 1;
  int howmany, sign;
  MPI_Datatype MPI_type_complex;

  this->_init();

  /* y corresponds to dim 0 in physical space */
  /* x corresponds to dim 1 in physical space */
  ny = N0;
  nx = N1;

  nX0 = N0;
  nX1 = N1;
  nX0loc = N0 / nb_proc;
  nXyloc = nX0loc;

  nKx = nx / 2;
  nKxloc = nKx / nb_proc;
  nKy = ny;

  /* This 2D fft is transposed */
  is_transposed = 1;
  nK0 = N1 / 2;
  nK0loc = nK0 / nb_proc;
  nK1 = N0;

  local_X0_start = rank * nX0loc;
  local_K0_start = rank * nK0loc;

  flags = FFTW_MEASURE;
  /*    flags = FFTW_ESTIMATE;*/
  /*    flags = FFTW_PATIENT;*/

  arrayX = (myreal *)fftw_malloc(sizeof(myreal) * nX0loc * N1);
  arrayK_pR = (mycomplex *)fftw_malloc(sizeof(mycomplex) * nX0loc * (nKx + 1));
  arrayK_pC = (mycomplex *)fftw_malloc(sizeof(mycomplex) * nKxloc * N0);

  auto start_time = chrono::high_resolution_clock::now();

  howmany = nX0loc;
  plan_r2c =
      fftw_plan_many_dft_r2c(1, &N1, howmany, arrayX, NULL, istride, N1,
                             reinterpret_cast<mycomplex_fftw *>(arrayK_pR),
                             NULL, ostride, nKx + 1, flags);

  plan_c2r = fftw_plan_many_dft_c2r(
      1, &N1, howmany, reinterpret_cast<mycomplex_fftw *>(arrayK_pR), NULL,
      istride, nKx + 1, arrayX, NULL, ostride, N1, flags);

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

  auto end_time = chrono::high_resolution_clock::now();

  clocktime_in_sec = end_time - start_time;

  if (rank == 0)
    printf("Initialization (%s) done in %f s\n", this->get_classname(),
           clocktime_in_sec.count());

  for (iX0 = 0; iX0 < nX0loc; iX0++)
    arrayK_pR[iX0 * (nKx + 1) + nKx] = 0.;

  MPI_Type_contiguous(2, MPI_DOUBLE, &MPI_type_complex);
  MPI_Type_commit(&MPI_type_complex);

  MPI_Type_vector(nX0loc, 1, nKx + 1, MPI_type_complex, &(MPI_type_column));
  MPI_Type_create_resized(MPI_type_column, 0, sizeof(mycomplex),
                          &(MPI_type_column));
  MPI_Type_commit(&(MPI_type_column));

  MPI_Type_vector(nKxloc, nX0loc, N0, MPI_type_complex, &(MPI_type_block));
  MPI_Type_create_resized(MPI_type_block, 0, nX0loc * sizeof(mycomplex),
                          &(MPI_type_block));
  MPI_Type_commit(&(MPI_type_block));
}

void FFT2DMPIWithFFTW1D::destroy(void) {
  // cout << "Object is being destroyed" << endl;
  fftw_destroy_plan(plan_r2c);
  fftw_destroy_plan(plan_c2c_fwd);
  fftw_destroy_plan(plan_c2c_bwd);
  fftw_destroy_plan(plan_c2r);
  fftw_free(arrayX);
  fftw_free(arrayK_pR);
  fftw_free(arrayK_pC);
  MPI_Type_free(&(MPI_type_column));
  MPI_Type_free(&(MPI_type_block));
}

FFT2DMPIWithFFTW1D::~FFT2DMPIWithFFTW1D(void) {
  // cout << "Object is being deleted" << endl;
}

bool FFT2DMPIWithFFTW1D::are_parameters_bad() {
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

char const *FFT2DMPIWithFFTW1D::get_classname() { return "FFT2DMPIWithFFTW1D"; }

myreal FFT2DMPIWithFFTW1D::compute_energy_from_X(myreal *fieldX) {
  int i0, i1;
  myreal energy_loc = 0;
  myreal energy;

  for (i0 = 0; i0 < nX0loc; i0++)
    for (i1 = 0; i1 < nX1; i1++)
      energy_loc += pow(fieldX[i1 + i0 * nX1], 2);

  MPI_Allreduce(&energy_loc, &energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  return energy / 2 * inv_coef_norm;
}

myreal FFT2DMPIWithFFTW1D::compute_energy_from_K(mycomplex *fieldK) {
  int i0, i1;
  myreal energy_loc = 0;
  myreal energy_tmp = 0;
  myreal energy;

  // modes i0 = iKx = 0
  i0 = 0;
  for (i1 = 0; i1 < nK1; i1++)
    energy_tmp += pow(abs(fieldK[i1]), 2);

  if (rank == 0) // i.e. if iKx == 0
    energy_loc = energy_tmp / 2;
  else
    energy_loc = energy_tmp;

  // other modes
  for (i0 = 1; i0 < nK0loc; i0++)
    for (i1 = 0; i1 < nK1; i1++)
      energy_loc += pow(abs(fieldK[i1 + i0 * nK1]), 2);

  MPI_Allreduce(&energy_loc, &energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  return energy;
}

myreal FFT2DMPIWithFFTW1D::sum_wavenumbers(myreal *fieldK) {
  int i0, i1;
  myreal sum_loc = 0;
  myreal sum_tmp = 0;
  myreal sum_tot;

  // modes i0 = iKx = 0
  i0 = 0;
  for (i1 = 0; i1 < nK1; i1++)
    sum_tmp += fieldK[i1];

  if (rank == 0) // i.e. if iKx == 0
    sum_loc = sum_tmp / 2;
  else
    sum_loc = sum_tmp;

  // other modes
  for (i0 = 1; i0 < nK0loc; i0++)
    for (i1 = 0; i1 < nK1; i1++)
      sum_loc += fieldK[i1 + i0 * nK1];

  MPI_Allreduce(&sum_loc, &sum_tot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  return 2 * sum_tot;
}

myreal FFT2DMPIWithFFTW1D::compute_mean_from_X(myreal *fieldX) {
  myreal mean, local_mean;
  int ii;
  local_mean = 0.;

  for (ii = 0; ii < nX0loc * nX1; ii++)
    local_mean += fieldX[ii];

  MPI_Allreduce(&local_mean, &mean, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  return mean * inv_coef_norm;
}

myreal FFT2DMPIWithFFTW1D::compute_mean_from_K(mycomplex *fieldK) {
  myreal mean;
  if (rank == 0)
    mean = real(fieldK[0]);

  MPI_Bcast(&mean, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  return mean;
}

void FFT2DMPIWithFFTW1D::fft(myreal *fieldX, mycomplex *fieldK) {
  int ii;
  // cout << "FFT2DMPIWithFFTW1D::fft" << endl;
  /*use memcpy(void * destination, void * source, size_t bytes); */
  memcpy(arrayX, fieldX, nX0loc * nX1 * sizeof(myreal));

  fftw_execute(plan_r2c);

  MPI_Alltoall(arrayK_pR, nKxloc, MPI_type_column, arrayK_pC, 1, MPI_type_block,
               MPI_COMM_WORLD);

  fftw_execute(plan_c2c_fwd);

  for (ii = 0; ii < nKxloc * nKy; ii++)
    fieldK[ii] = arrayK_pC[ii] * inv_coef_norm;
}

void FFT2DMPIWithFFTW1D::ifft(mycomplex *fieldK, myreal *fieldX) {
  int ii;
  // cout << "FFT2DMPIWithFFTW1D::ifft" << endl;
  memcpy(arrayK_pC, fieldK, nKxloc * nKy * sizeof(mycomplex));
  fftw_execute(plan_c2c_bwd);
  MPI_Alltoall(arrayK_pC, 1, MPI_type_block, arrayK_pR, nKxloc, MPI_type_column,
               MPI_COMM_WORLD);

  /*These modes (nx/2+1=N1/2+1) have to be settled to zero*/
  for (ii = 0; ii < nX0loc; ++ii)
    arrayK_pR[ii * (nKx + 1) + nKx] = 0.;

  fftw_execute(plan_c2r);
  memcpy(fieldX, arrayX, nX0loc * nX1 * sizeof(myreal));
}
