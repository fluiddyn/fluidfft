
#include <iostream>
using namespace std;

#include <base_fft2d.h>

BaseFFT2D::BaseFFT2D(int argN0, int argN1) {
  N0 = argN0;
  N1 = argN1;
  is_transposed = 0;
  local_X0_start = 0;
  local_K0_start = 0;
  inv_coef_norm = 1. / N0;
  inv_coef_norm = inv_coef_norm / N1;
}

void BaseFFT2D::_init() {
  BaseFFT::_init();
  if (rank == 0)
    cout << "N0 = " << N0 << " ; N1 = " << N1 << endl;
}

int BaseFFT2D::get_local_size_X() { return nX0loc * nX1; }

int BaseFFT2D::get_local_size_K() { return nKxloc * nKy; }

void BaseFFT2D::get_local_shape_X(int *ptr_nX0loc, int *ptr_nX1) {
  *ptr_nX0loc = nX0loc;
  *ptr_nX1 = nX1;
}

void BaseFFT2D::get_local_shape_K(int *ptr_nK0loc, int *ptr_nK1) {
  *ptr_nK0loc = nK0loc;
  *ptr_nK1 = nK1;
}

char BaseFFT2D::get_is_transposed() { return is_transposed; }

ptrdiff_t BaseFFT2D::get_local_X0_start() { return local_X0_start; }

ptrdiff_t BaseFFT2D::get_local_K0_start() { return local_K0_start; }

void BaseFFT2D::get_shapeX_seq(int *ptr_nX0, int *ptr_nX1) {
  *ptr_nX0 = nX0;
  *ptr_nX1 = nX1;
}

void BaseFFT2D::get_shapeK_seq(int *ptr_nK0, int *ptr_nK1) {
  *ptr_nK0 = nK0;
  *ptr_nK1 = nK1;
}
