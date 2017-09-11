
from __future__ import print_function, division

from time import time

import numpy as np

from fluiddyn.util import mpi

from fluidfft.fft2d import get_classes_seq, get_classes_mpi


rank = mpi.rank
nb_proc = mpi.nb_proc


print_old = print


def print(*args, **kwargs):
    if mpi.rank == 0:
        print_old(*args, **kwargs)


def bench_like_cpp_as_arg(o, nb_time_execute=10):

    fieldX = np.ones(o.get_shapeX_loc(), dtype=float, order='C')
    fieldK = np.empty(o.get_shapeK_loc(), dtype=np.complex128, order='C')

    t_start = time()
    for i in range(nb_time_execute):
        fieldK = o.fft_as_arg(fieldX, fieldK)
    t_end = time()
    t_fft = (t_end - t_start)/nb_time_execute
    print('time fft_as_arg:  {}'.format(t_fft))

    t_start = time()
    for i in range(nb_time_execute):
        fieldX = o.ifft_as_arg(fieldK, fieldX)
    t_end = time()
    t_ifft = (t_end - t_start)/nb_time_execute
    print('time ifft_as_arg: {}'.format(t_ifft))
    return t_fft, t_ifft


def bench_like_cpp_return(o, nb_time_execute=10):

    fieldX = np.ones(o.get_shapeX_loc(), dtype=float, order='C')
    fieldK = np.empty(o.get_shapeK_loc(), dtype=np.complex128, order='C')

    t_start = time()
    for i in range(nb_time_execute):
        o.fft(fieldX)
    t_end = time()
    t_fft = (t_end - t_start)/nb_time_execute
    print('time return_fft:  {}'.format(t_fft))

    t_start = time()
    for i in range(nb_time_execute):
        o.ifft(fieldK)
    t_end = time()
    t_ifft = (t_end - t_start)/nb_time_execute
    print('time return_ifft: {}'.format(t_ifft))
    return t_fft, t_ifft


def pourc(t_slow, t_fast):
    return 100. * (t_slow - t_fast) / t_fast


def compare_benchs(o, nb_time_execute=10):

    results = o.run_benchs(nb_time_execute)

    t_fft_as_arg, t_ifft_as_arg = bench_like_cpp_as_arg(o, nb_time_execute)
    t_fft_return, t_ifft_return = bench_like_cpp_return(o, nb_time_execute)

    if results:
        t_fft_cpp, t_ifft_cpp = results

        txt = 'fft is {:4.2f} % slower than cpp'
        print('as_arg ' + txt.format(pourc(t_fft_as_arg, t_fft_cpp)))
        print('as_arg i' + txt.format(pourc(t_ifft_as_arg, t_ifft_cpp)))
        print('return ' + txt.format(pourc(t_fft_return, t_fft_cpp)))
        print('return i' + txt.format(pourc(t_ifft_return, t_ifft_cpp)))


if __name__ == '__main__':

    n = 1024 * 2  # / 4

    def run(FFT2D):
        if FFT2D is None:
            return
        o = FFT2D(n, n)
        o.run_tests()
        o.run_benchs()
        compare_benchs(o, nb_time_execute=10)

    if rank == 0:
        for FFT2D in get_classes_seq().values():
            run(FFT2D)

    if nb_proc > 1:
        for FFT2D in get_classes_mpi().values():
            run(FFT2D)