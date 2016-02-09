
import numpy as np

from fluidfft3d.with_fftw3d import FFT3DWithFFTW3D as FFT3D

from fluidfft2d.with_fftw2d import FFT2DWithFFTW2D as FFT2D


def build_arrayX_from_2d_indices12(self, o2d, arr2d):

    nX0, nX1, nX2 = self.get_shapeX_seq()
    nX0loc, nX1loc, nX2loc = self.get_shapeX_loc()

    if (nX1, nX2) != o2d.get_shapeX_seq():
        raise ValueError('Not the same physical shape...')

    # check that the 2d fft is not with distributed memory...
    if o2d.get_shapeX_loc() != o2d.get_shapeX_loc():
        raise ValueError('2d fft is with distributed memory...')

    ind0seq_first, ind1seq_first = self.get_seq_indices_first_K()

    if (nX1loc, nX2loc) == o2d.get_shapeX_loc():
        arr3d_loc_2dslice = arr2d
    else:
        raise NotImplementedError

    arr3d = np.empty([nX0loc, nX1loc, nX2loc])
    for i0 in range(nX0loc):
        arr3d[i0] = arr3d_loc_2dslice

    return arr3d

n = 8

n0 = n
n1 = 2*n
n2 = 3*n

o3d = FFT3D(n0, n1, n2)
o2d = FFT2D(n1, n2)

shapeX_loc_2d = o2d.get_shapeX_loc()

arr2d = np.arange(np.product(shapeX_loc_2d)).reshape(o2d.get_shapeX_loc())

# arr3d = build_arrayX_from_2d_indices12(o3d, o2d, arr2d)
arr3d = o3d.build_arrayX_from_2d_indices12(o2d, arr2d)
