
#include <iostream>
using namespace std;

#include <base_fft3d.h>


BaseFFT3D::BaseFFT3D(int argN0, int argN1, int argN2)
{
  N0 = argN0;
  N1 = argN1;
  N2 = argN2;
}


void BaseFFT3D::_init()
{
  BaseFFT::_init();
  if (rank == 0)
    cout << "N0 = " << N0 << " ; N1 = " << N1 << " ; N2 = " << N2 << endl;
}


int BaseFFT3D::get_local_size_X()
{
  return nX0loc * nX1loc * nX2;
}


int BaseFFT3D::get_local_size_K()
{
  return nK0loc * nK1loc * nK2;
}


void BaseFFT3D::get_local_shape_X(int *ptr_nX0loc, int *ptr_nX1loc,
				  int *ptr_nX2)
{
  *ptr_nX0loc = nX0loc;
  *ptr_nX1loc = nX1loc;
  *ptr_nX2 = nX2;
}


void BaseFFT3D::get_local_shape_K(int *ptr_nK0loc, int *ptr_nK1loc,
				  int *ptr_nK2)
{
  *ptr_nK0loc = nK0loc;
  *ptr_nK1loc = nK1loc;
  *ptr_nK2 = nK2;
}
