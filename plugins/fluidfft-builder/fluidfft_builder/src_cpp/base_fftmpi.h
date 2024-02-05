

#ifndef _CLASS_BaseFFTMPI
#define _CLASS_BaseFFTMPI

#include <mpi.h>

#include <base_fft.h>

class BaseFFTMPI : virtual public BaseFFT {
public:
  void _init_parallel();
};

#endif
