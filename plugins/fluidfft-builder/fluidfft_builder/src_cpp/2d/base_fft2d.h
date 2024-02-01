

#ifndef _CLASS_BaseFFT2D
#define _CLASS_BaseFFT2D

#include <base_fft.h>

class BaseFFT2D : virtual public BaseFFT {
public:
  BaseFFT2D(int N0, int N1);
  void _init();

  virtual int get_local_size_X();
  virtual int get_local_size_K();

  virtual void get_local_shape_X(int *, int *);
  virtual void get_local_shape_K(int *, int *);

  virtual void get_shapeX_seq(int *, int *);
  virtual void get_shapeK_seq(int *, int *);

  char get_is_transposed();
  ptrdiff_t get_local_X0_start();
  ptrdiff_t get_local_K0_start();

  char is_transposed;
  ptrdiff_t local_X0_start, local_K0_start;

  myreal inv_coef_norm;
};

#endif
