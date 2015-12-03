#include <stdlib.h>

#include <base_fft3dmpi.h>

BaseFFT3DMPI::BaseFFT3DMPI(int argN0, int argN1, int argN2):
  BaseFFT3D::BaseFFT3D(argN0, argN1, argN2)
{}


void calcul_nprocmesh(int rank, int nb_proc, int* nprocmesh)
{
  int np0, np1;
  char* split;
  char* PROC_MESH;
  
  PROC_MESH = getenv("FLUID_PROC_MESH");
  if (PROC_MESH!=NULL)
    {
      if (rank == 0)
	printf("FLUID_PROC_MESH: %s\n", PROC_MESH);
      split = strtok(PROC_MESH, "x");
      if (split != NULL)
	{
	  np0 = atoi(split);
	  split = strtok(NULL, "x");
	}
      else
	np0 = 2;

      if (split != NULL)
	np1 = atoi(split);
      else
	np1 = nb_proc / np0;
    }
  else
    {
      // find the most symmetric proc mesh: 
      np0 = sqrt(nb_proc);
      np1 = nb_proc / np0;
      while (np0*np1 != nb_proc and np0 != 1)
	{
	  np0--;
	  np1 = nb_proc / np0;
	}
    }
  
  nprocmesh[0] = np0;
  nprocmesh[1] = np1;

}
