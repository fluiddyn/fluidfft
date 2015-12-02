

#ifndef _CLASS_BaseFFT2D
#define _CLASS_BaseFFT2D


#include <base_fft.h>

class BaseFFT2D: virtual public BaseFFT
{
 public:
  BaseFFT2D(int N0, int N1);
  void _init();

  virtual int get_local_size_X();
  virtual int get_local_size_K();

  virtual void get_local_shape_X(int*, int*);
  virtual void get_local_shape_K(int*, int*);
};

#endif
