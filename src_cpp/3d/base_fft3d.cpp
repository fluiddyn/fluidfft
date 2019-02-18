
#include <iostream>
using namespace std;

#include <base_fft3d.h>

BaseFFT3D::BaseFFT3D(int argN0, int argN1, int argN2) {
  N0 = argN0;
  N1 = argN1;
  N2 = argN2;
}

void BaseFFT3D::_init() {
  BaseFFT::_init();
  if (rank == 0)
    cout << "N0 = " << N0 << " ; N1 = " << N1 << " ; N2 = " << N2 << endl;
  inv_coef_norm = 1. / N0;
  inv_coef_norm = inv_coef_norm / N1;
  inv_coef_norm = inv_coef_norm / N2;
}

int BaseFFT3D::get_local_size_X() { return nX0loc * nX1loc * nX2loc; }

int BaseFFT3D::get_local_size_K() { return nK0loc * nK1loc * nK2loc; }

void BaseFFT3D::get_local_shape_X(int *ptr_nX0loc, int *ptr_nX1loc,
                                  int *ptr_nX2loc) {
  *ptr_nX0loc = nX0loc;
  *ptr_nX1loc = nX1loc;
  *ptr_nX2loc = nX2loc;
}

void BaseFFT3D::get_local_shape_K(int *ptr_nK0loc, int *ptr_nK1loc,
                                  int *ptr_nK2loc) {
  *ptr_nK0loc = nK0loc;
  *ptr_nK1loc = nK1loc;
  *ptr_nK2loc = nK2loc;
}

void BaseFFT3D::get_global_shape_X(int *ptr_nX0, int *ptr_nX1, int *ptr_nX2) {
  *ptr_nX0 = nX0;
  *ptr_nX1 = nX1;
  *ptr_nX2 = nX2;
}

void BaseFFT3D::get_global_shape_K(int *ptr_nK0, int *ptr_nK1, int *ptr_nK2) {
  *ptr_nK0 = nK0;
  *ptr_nK1 = nK1;
  *ptr_nK2 = nK2;
}

void BaseFFT3D::get_dimX_K(int *d0, int *d1, int *d2) {
  *d0 = 0;
  *d1 = 1;
  *d2 = 2;
}

void BaseFFT3D::get_seq_indices_first_X(int *i0, int *i1, int *i2) {
  *i0 = 0;
  *i1 = 0;
  *i2 = 0;
}

void BaseFFT3D::get_seq_indices_first_K(int *i0, int *i1, int *i2) {
  *i0 = 0;
  *i1 = 0;
  *i2 = 0;
}

myreal BaseFFT3D::compute_mean_from_X(myreal *fieldX) {
  myreal mean, mean1, mean2;
  int ii, jj, kk;
  mean = 0.;

  for (ii = 0; ii < nX0; ii++) {
    mean1 = 0.;
    for (jj = 0; jj < nX1; jj++) {
      mean2 = 0.;
      for (kk = 0; kk < nX2; kk++)
        mean2 += fieldX[(ii * nX1 + jj) * nX2 + kk];
      mean1 += mean2 / nX2;
    }
    mean += mean1 / nX1;
  }
  return mean / nX0;
}
