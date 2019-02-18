
#ifndef _CLASS_BaseFFT3DMPI
#define _CLASS_BaseFFT3DMPI

#include <base_fft3d.h>
#include <base_fftmpi.h>

class BaseFFT3DMPI : public BaseFFT3D, public BaseFFTMPI {
public:
  BaseFFT3DMPI(int N0, int N1, int N2);

  virtual myreal compute_mean_from_K(mycomplex *fieldK);
  virtual myreal compute_mean_from_X(myreal *fieldX);
  virtual myreal compute_energy_from_X(myreal *fieldX);

protected:
  int nX2_pad, nXzloc;
  int nKxloc, nKyloc, nKzloc;
};

void calcul_nprocmesh(int rank, int nb_proc, int *nprocmesh);

#endif
