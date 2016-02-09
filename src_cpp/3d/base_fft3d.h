

#ifndef _CLASS_BaseFFT3D
#define _CLASS_BaseFFT3D


#include <base_fft.h>

class BaseFFT3D: virtual public BaseFFT
{
 public:
  BaseFFT3D(int N0, int N1, int N2);
  void _init();

  virtual int get_local_size_X();
  virtual int get_local_size_K();

  virtual void get_local_shape_X(int*, int*, int*);
  virtual void get_local_shape_K(int*, int*, int*);

  virtual void get_global_shape_X(int*, int*, int*);
  virtual void get_global_shape_K(int*, int*, int*);

  virtual void get_dimX_K(int*, int*, int*);
  virtual void get_seq_indices_first_K(int*, int*);
  
 protected:
  int N2, nz, nX2, nK2, nK1loc, nKz, nXz, nX1loc;
  
};

#endif
