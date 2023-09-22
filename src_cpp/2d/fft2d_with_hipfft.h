
/* #include <stdio.h> */
#include <base_fft2d.h>
#include <hip/hip_runtime.h>
#include <hipfft.h>

#ifdef SINGLE_PREC
typedef float2 dcomplex;
#else
typedef double2 dcomplex;
#endif

class FFT2DWithHIPFFT : public BaseFFT2D {
public:
  FFT2DWithHIPFFT(int N0, int N1);
  ~FFT2DWithHIPFFT();
  void destroy();

  void fft(myreal *fieldX, mycomplex *fieldK);
  void ifft(mycomplex *fieldK, myreal *fieldX);
  myreal compute_energy_from_K(mycomplex *fieldK);
  myreal sum_wavenumbers(myreal *fieldK);
  myreal compute_mean_from_K(mycomplex *fieldK);

  virtual const char *get_classname();

  myreal compute_energy_from_X(myreal *fieldX);
  myreal compute_mean_from_X(myreal *fieldX);

private:
  int nX1loc, nK1loc, nXxloc, nXyloc, nXx, nXy, nKyloc;

  int mem_size;  // equivalent à la taille de arrayK?
  int mem_sizer; // equivalent à la taille de arrayK?

  // Allocate device memory for signal
  myreal *arrayX;
  myreal *arrayK;
  dcomplex *data;
  myreal *datar;
  hipfftHandle plan;
  hipfftHandle plan1;
};
