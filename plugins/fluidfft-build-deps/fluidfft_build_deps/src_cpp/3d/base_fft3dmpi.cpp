#include <stdlib.h>

#include <base_fft3dmpi.h>

BaseFFT3DMPI::BaseFFT3DMPI(int argN0, int argN1, int argN2)
    : BaseFFT3D::BaseFFT3D(argN0, argN1, argN2) {}

void calcul_nprocmesh(int rank, int nb_proc, int *nprocmesh) {
  int np0, np1;
  char *split;
  char *PROC_MESH;

  PROC_MESH = getenv("FLUID_PROC_MESH");
  if (PROC_MESH != NULL) {
    if (rank == 0)
      printf("FLUID_PROC_MESH: %s\n", PROC_MESH);
    split = strtok(PROC_MESH, "x");
    if (split != NULL) {
      np0 = atoi(split);
      split = strtok(NULL, "x");
    } else
      np0 = 2;

    if (split != NULL)
      np1 = atoi(split);
    else
      np1 = nb_proc / np0;
  } else {
    // find the most symmetric proc mesh:
    np0 = sqrt(nb_proc);
    np1 = nb_proc / np0;
    while (np0 * np1 != nb_proc and np0 != 1) {
      np0--;
      np1 = nb_proc / np0;
    }
  }

  nprocmesh[0] = np0;
  nprocmesh[1] = np1;
}

myreal BaseFFT3DMPI::compute_mean_from_X(myreal *fieldX) {
  myreal mean, local_mean;
  int ii;
  local_mean = 0.;

  for (ii = 0; ii < nX0loc * nX1loc * nX2loc; ii++)
    local_mean += fieldX[ii];
#ifdef SINGLE_PREC
  MPI_Allreduce(&local_mean, &mean, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
#else
  MPI_Allreduce(&local_mean, &mean, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  return mean * inv_coef_norm;
}

myreal BaseFFT3DMPI::compute_mean_from_K(mycomplex *fieldK) {
  myreal mean;
  if (rank == 0)
    mean = real(fieldK[0]);

#ifdef SINGLE_PREC
  MPI_Bcast(&mean, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
#else
  MPI_Bcast(&mean, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
  return mean;
}

myreal BaseFFT3DMPI::compute_energy_from_X(myreal *fieldX) {
  int ii;
  myreal energy_loc = 0;
  myreal energy;

  for (ii = 0; ii < nX0loc * nX1loc * nX2loc; ii++)
    energy_loc += (double)pow(fieldX[ii], 2);
#ifdef SINGLE_PREC
  MPI_Allreduce(&energy_loc, &energy, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
#else
  MPI_Allreduce(&energy_loc, &energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  return (myreal)energy / 2 * inv_coef_norm;
}
