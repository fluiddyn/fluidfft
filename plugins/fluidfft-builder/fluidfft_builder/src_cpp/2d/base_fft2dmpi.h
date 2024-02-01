
#ifndef _CLASS_BaseFFT2DMPI
#define _CLASS_BaseFFT2DMPI

#include <base_fft2d.h>
#include <base_fftmpi.h>

class BaseFFT2DMPI : public BaseFFT2D, public BaseFFTMPI {
public:
  BaseFFT2DMPI(int N0, int N1);
};

#endif
