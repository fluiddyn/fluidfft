
#include <fft3dmpi_with_p3dfft.h>

FFT3DMPIWithP3DFFT::FFT3DMPIWithP3DFFT(int argN0, int argN1, int argN2)
    : BaseFFT3DMPI::BaseFFT3DMPI(argN0, argN1, argN2) {
  chrono::duration<double> clocktime_in_sec;

  int conf;
  int memsize[3];
  int istart[3], isize[3], iend[3];
  int fstart[3], fsize[3], fend[3];

  this->_init();

  nz = N0;
  ny = N1;
  nx = N2;
  nXz = N0;
  nXy = N1;
  nXx = N2;

  nX0 = N0;
  nX1 = N1;
  nX2 = N2;

  calcul_nprocmesh(rank, nb_proc, nprocmesh);

  auto start_time = chrono::high_resolution_clock::now();

  if (rank == 0)
    printf("Using processor grid %d x %d\n", nprocmesh[0], nprocmesh[1]);

  /* Initialize P3DFFT */
  Cp3dfft_setup(nprocmesh, N2, N1, N0, MPI_Comm_c2f(MPI_COMM_WORLD), nX2, nX1,
                nX0, 0, memsize);
  /* Get dimensions for input array - real numbers, X-pencil shape.
   *    *       Note that we are following the Fortran ordering, i.e.
   *       *             the dimension with stride-1 is X. */
  conf = 1;
  Cp3dfft_get_dims(istart, iend, isize, conf);
  /* Get dimensions for output array - complex numbers, Z-pencil shape.
   *    *       Stride-1 dimension could be X or Z, depending on how the library
   *       *             was compiled (stride1 option) */
  conf = 2;
  Cp3dfft_get_dims(fstart, fend, fsize, conf);

  /* in physical space: */
  /* z corresponds to dim 0 */
  /* y corresponds to dim 1 */
  /* x corresponds to dim 2 */
  nX0loc = isize[2];
  nXzloc = nX0loc;
  nX1loc = isize[1];
  nXyloc = nX1loc;
  nX2loc = isize[0];
  nXxloc = nX2loc;

  /* This 3D fft is NOT transposed (??) */
  /* in Fourier space: */
  /* kz corresponds to dim 0 */
  /* ky corresponds to dim 1 */
  /* kx corresponds to dim 2 */
  nKx = nx / 2 + 1;
  nKy = ny;
  nKz = nz;

  nK0 = nKz;
  nK1 = nKy;
  nK2 = nKx;

  // Warning: order as in X space!
  nKzloc = fsize[2];
  nKyloc = fsize[1];
  nKxloc = fsize[0];

  nK0loc = nKzloc;
  nK1loc = nKyloc;
  nK2loc = nKxloc;
  MPI_Barrier(MPI_COMM_WORLD);
  fflush(stdout);

  MPI_Barrier(MPI_COMM_WORLD);
  local_X0_start = istart[2] - 1;
  local_X1_start = istart[1] - 1;
  local_X2_start = istart[0] - 1;
  // Warning: order as in X space!
  local_K0_start = fstart[2] - 1;
  local_K1_start = fstart[1] - 1;
  local_K2_start = fstart[0] - 1;

  auto end_time = chrono::high_resolution_clock::now();

  clocktime_in_sec = end_time - start_time;

  if (rank == 0)
    printf("Initialization (%s) done in %f s\n", this->get_classname(),
           clocktime_in_sec.count());
}

void FFT3DMPIWithP3DFFT::destroy(void) {
  //   cout << "Object is being destroyed" << endl;
  // Cp3dfft_clean();
}

FFT3DMPIWithP3DFFT::~FFT3DMPIWithP3DFFT(void) {}

char const *FFT3DMPIWithP3DFFT::get_classname() { return "FFT3DMPIWithP3DFFT"; }

myreal FFT3DMPIWithP3DFFT::compute_energy_from_K(mycomplex *fieldK) {
  int i0, i1, i2;
  double energy_tmp = 0;
  double energy_loc = 0;
  double energy;
  i2 = 0;
  energy_tmp = 0.;
  for (i0 = 0; i0 < nK0loc; i0++)
    for (i1 = 0; i1 < nK1loc; i1++)
      energy_tmp +=
          (double)square_abs(fieldK[i2 + (i1 + i0 * nK1loc) * nK2loc]);

  if ((local_K2_start == 0) || (nK2loc == 1 and local_K2_start + nK2loc == nK2))
    energy_tmp /= 2.;
  energy_loc = energy_tmp;

  if (nK2loc > 1) {
    //  modes i1_seq iKx = last = nK1 - 1
    i2 = nK2loc - 1;
    energy_tmp = 0;
    for (i0 = 0; i0 < nK0loc; i0++)
      for (i1 = 0; i1 < nK1loc; i1++)
        energy_tmp +=
            (double)square_abs(fieldK[i2 + (i1 + i0 * nK1loc) * nK2loc]);

    if (local_K2_start + nK2loc == nK2)
      energy_tmp /= 2.;
    energy_loc += energy_tmp;

    for (i0 = 0; i0 < nK0loc; i0++)
      for (i1 = 0; i1 < nK1loc; i1++)
        for (i2 = 1; i2 < nK2loc - 1; i2++)
          energy_loc +=
              (double)square_abs(fieldK[i2 + (i1 + i0 * nK1loc) * nK2loc]);
  }

  // case nK1loc == 0
  if (min(nK0loc, nK2loc) == 0) {
    energy_loc = 0;
  }

  MPI_Allreduce(&energy_loc, &energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  return (myreal)energy;
}

myreal FFT3DMPIWithP3DFFT::sum_wavenumbers_double(myreal *fieldK) {
  int i0, i1, i2;
  double sum_tmp = 0;
  double sum_loc = 0;
  double sum;
  i2 = 0;
  sum_tmp = 0.;
  for (i0 = 0; i0 < nK0loc; i0++)
    for (i1 = 0; i1 < nK1loc; i1++)
      sum_tmp += (double)fieldK[i2 + (i1 + i0 * nK1loc) * nK2loc];

  if ((local_K2_start == 0) || (nK2loc == 1 and local_K2_start + nK2loc == nK2))
    sum_tmp /= 2.;
  sum_loc = sum_tmp;

  if (nK2loc != 1) {
    //  modes i1_seq iKx = last = nK1 - 1
    i2 = nK2loc - 1;
    sum_tmp = 0;
    for (i0 = 0; i0 < nK0loc; i0++)
      for (i1 = 0; i1 < nK1loc; i1++)
        sum_tmp += (double)fieldK[i2 + (i1 + i0 * nK1loc) * nK2loc];

    if (local_K2_start + nK2loc == nK2)
      sum_tmp /= 2.;
    sum_loc += sum_tmp;

    for (i0 = 0; i0 < nK0loc; i0++)
      for (i1 = 0; i1 < nK1loc; i1++)
        for (i2 = 1; i2 < nK2loc - 1; i2++)
          sum_loc += (double)fieldK[i2 + (i1 + i0 * nK1loc) * nK2loc];
  }

  // case nK1loc==0
  if (min(nK0loc, nK2loc) == 0) {
    sum_loc = 0;
  }

  MPI_Allreduce(&sum_loc, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  sum *= 2.;

  return (myreal)sum;
}

void FFT3DMPIWithP3DFFT::sum_wavenumbers_complex(mycomplex *fieldK,
                                                 mycomplex *result) {
  mycomplex sum_tmp = 0;
  mycomplex sum_loc = 0;
  mycomplex sum;

  int i0, i1, i2;

  i2 = 0;
  for (i0 = 0; i0 < nK0loc; i0++)
    for (i1 = 0; i1 < nK1loc; i1++)
      sum_tmp += fieldK[i2 + (i1 + i0 * nK1loc) * nK2loc];

  if ((local_K2_start == 0) || (nK2loc == 1 and local_K2_start + nK2loc == nK2))
    sum_tmp /= 2.;
  sum_loc = sum_tmp;

  if (nK2loc != 1) {
    //  modes i1_seq iKx = last = nK1 - 1
    i2 = nK2loc - 1;
    sum_tmp = 0;
    for (i0 = 0; i0 < nK0loc; i0++)
      for (i1 = 0; i1 < nK1loc; i1++)
        sum_tmp += fieldK[i2 + (i1 + i0 * nK1loc) * nK2loc];

    if (local_K2_start + nK2loc == nK2)
      sum_tmp /= 2.;
    sum_loc += sum_tmp;

    for (i0 = 0; i0 < nK0loc; i0++)
      for (i1 = 0; i1 < nK1loc; i1++)
        for (i2 = 1; i2 < nK2loc - 1; i2++)
          sum_loc += fieldK[i2 + (i1 + i0 * nK1loc) * nK2loc];
  }

  // case nK1loc==0
  if (min(nK0loc, nK2loc) == 0) {
    sum_loc = 0;
  }

  MPI_Allreduce(&sum_loc, &sum, 1, MPI_COMPLEX, MPI_SUM, MPI_COMM_WORLD);

  sum *= 2.;
  *result = sum;
}

myreal FFT3DMPIWithP3DFFT::compute_mean_from_K(mycomplex *fieldK) {
  double mean, local_mean;

  if (local_K2_start == 0 and local_K1_start == 0)
    local_mean = (double)real(fieldK[0]);
  else
    local_mean = 0.;

  MPI_Allreduce(&local_mean, &mean, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  return (myreal)mean;
}

void FFT3DMPIWithP3DFFT::fft(myreal *fieldX, mycomplex *fieldK) {
  int i;
  unsigned char op_f[] = "fft";
  // cout << "FFT3DMPIWithP3DFFT::fft" << endl;

  Cp3dfft_ftran_r2c(reinterpret_cast<myreal *>(fieldX),
                    reinterpret_cast<myreal *>(fieldK), op_f);

  for (i = 0; i < nK0loc * nK1loc * nK2loc; i++)
    fieldK[i] *= inv_coef_norm;
}

void FFT3DMPIWithP3DFFT::ifft(mycomplex *fieldK, myreal *fieldX) {
  unsigned char op_b[] = "tff";
  // cout << "FFT3DMPIWithP3DFFT::ifft" << endl;

  Cp3dfft_btran_c2r(reinterpret_cast<myreal *>(fieldK),
                    reinterpret_cast<myreal *>(fieldX), op_b);
}

void FFT3DMPIWithP3DFFT::ifft_destroy(mycomplex *fieldK, myreal *fieldX) {
  unsigned char op_b[] = "tff";
  Cp3dfft_btran_c2r(reinterpret_cast<myreal *>(fieldK),
                    reinterpret_cast<myreal *>(fieldX), op_b);
}

void FFT3DMPIWithP3DFFT::get_dimX_K(int *d0, int *d1, int *d2) {
  // As in sequential! Not transposed!
  *d0 = 0;
  *d1 = 1;
  *d2 = 2;
}

void FFT3DMPIWithP3DFFT::get_seq_indices_first_K(int *i0, int *i1, int *i2) {
  *i0 = local_K0_start;
  *i1 = local_K1_start;
  *i2 = local_K2_start;
}

void FFT3DMPIWithP3DFFT::get_seq_indices_first_X(int *i0, int *i1, int *i2) {
  *i0 = local_X0_start;
  *i1 = local_X1_start;
  *i2 = local_X2_start;
}

bool FFT3DMPIWithP3DFFT::are_parameters_bad() {
  calcul_nprocmesh(rank, nb_proc, nprocmesh);
  if (N2 <= 1 || N1 < nprocmesh[1] || (N0 * N1) / nb_proc == 0) {
    if (rank == 0)
      cout << "bad parameters N0=" << N0 << " or N1=" << N1 << " or N2=" << N2
           << endl;
    return 1;
  }
  return 0;
}
