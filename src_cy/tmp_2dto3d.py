
import numpy as np

from fluidfft import import_fft_class

FFT3D = import_fft_class('fft3d.mpi_with_fftwmpi3d')
FFT2D = import_fft_class('fft2d.with_fftw2d')


def build_arrayX_from_2d_indices12(self, o2d, arr2d):

    nX0, nX1, nX2 = self.get_shapeX_seq()
    nX0loc, nX1loc, nX2loc = self.get_shapeX_loc()

    if (nX1, nX2) != o2d.get_shapeX_seq():
        raise ValueError('Not the same physical shape...')

    # check that the 2d fft is not with distributed memory...
    if o2d.get_shapeX_loc() != o2d.get_shapeX_loc():
        raise ValueError('2d fft is with distributed memory...')

    if (nX1loc, nX2loc) == o2d.get_shapeX_loc():
        arr3d_loc_2dslice = arr2d
    else:
        raise NotImplementedError

    arr3d = np.empty([nX0loc, nX1loc, nX2loc])
    for i0 in range(nX0loc):
        arr3d[i0] = arr3d_loc_2dslice

    return arr3d


def build_invariant_arrayK_from_2d_indices12X(self, o2d, arr2d):

    nK0, nK1, nK2 = self.get_shapeK_seq()
    nK0loc, nK1loc, nK2loc = self.get_shapeK_loc()

    nX0, nX1, nX2 = self.get_shapeX_seq()

    if (nX1, nX2) != o2d.get_shapeX_seq():
        raise ValueError('Not the same physical shape...')

    # check that the 2d fft is not with distributed memory...
    if o2d.get_shapeX_loc() != o2d.get_shapeX_loc():
        raise ValueError('2d fft is with distributed memory...')

    ind0seq_first, ind1seq_first = self.get_seq_indices_first_K()
    dimX_K = self.get_dimX_K()

    arr3d = np.zeros([nK0loc, nK1loc, nK2loc], dtype=np.complex128)

    if dimX_K == (0, 1, 2):
        # simple
        if (nK0, nK1, nK2) == (nK0loc, nK1loc, nK2loc):
            # very simple
            arr3d_loc_2dslice = arr2d
        else:
            raise NotImplementedError

        arr3d[0] = arr3d_loc_2dslice

    elif dimX_K == (1, 0, 2):
        # like fft3d.mpi_with_fftwmpi3d
        arr3d_loc_2dslice = np.zeros([nK0loc, nK2loc], dtype=np.complex128)

        for i0 in range(nK0loc):
            for i2 in range(nK2loc):
                i0_2d = ind0seq_first + i0
                i1_2d = i2
                arr3d_loc_2dslice[i0, i2] = arr2d[i0_2d, i1_2d]

        arr3d[:, 0, :] = arr3d_loc_2dslice
    else:
        raise NotImplementedError

    return arr3d


n = 4

n0 = n
n1 = 10*n
n2 = 100*n

o3d = FFT3D(n0, n1, n2)
o2d = FFT2D(n1, n2)

shapeK_loc_2d = o2d.get_shapeK_loc()

arr2d = np.arange(np.product(shapeK_loc_2d)).reshape(
    shapeK_loc_2d).astype(np.complex128)

arr3d = build_invariant_arrayK_from_2d_indices12X(o3d, o2d, arr2d)
# arr3d = o3d.build_invariant_arrayK_from_2d_indices12X(o2d, arr2d)
