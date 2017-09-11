
from __future__ import print_function, division

from time import time

import numpy as np

from fluiddyn.util import mpi

from fluidfft.fft3d import get_classes_seq, get_classes_mpi

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
        o.fft_as_arg(fieldX, fieldK)
    t_end = time()
    t_fft = (t_end - t_start)/nb_time_execute
    print('time fft_as_arg:  {}'.format(t_fft))

    t_start = time()
    for i in range(nb_time_execute):
        o.ifft_as_arg(fieldK, fieldX)
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


def bench_like_cpp(o, nb_time_execute=10):
    fieldX = np.ones(o.get_shapeX_loc(), dtype=float, order='C')
    fieldK = np.empty(o.get_shapeK_loc(), dtype=np.complex128, order='C')

    t_start = time()
    for i in range(nb_time_execute):
        o.fft_generic(fieldX)
    t_end = time()
    t_fft = (t_end - t_start)/nb_time_execute
    print('time fft (generic):  {}'.format(t_fft))

    t_start = time()
    for i in range(nb_time_execute):
        o.ifft_generic(fieldK)
    t_end = time()
    t_ifft = (t_end - t_start)/nb_time_execute
    print('time ifft (generic): {}'.format(t_ifft))
    return t_fft, t_ifft


def pourc(t_slow, t_fast):
    return 100. * (t_slow - t_fast) / t_fast


def compare_benchs(o, nb_time_execute=10):

    results = o.run_benchs(nb_time_execute)

    t_fft_as_arg, t_ifft_as_arg = bench_like_cpp_as_arg(o, nb_time_execute)
    t_fft_return, t_ifft_return = bench_like_cpp_return(o, nb_time_execute)
    # t_fft_generic, t_ifft_generic = bench_like_cpp(o, nb_time_execute)

    if results:
        t_fft_cpp, t_ifft_cpp = results

        txt = 'fft is {:4.2f} % slower than cpp'
        print('as_arg ' + txt.format(pourc(t_fft_as_arg, t_fft_cpp)))
        print('as_arg i' + txt.format(pourc(t_ifft_as_arg, t_ifft_cpp)))
        print('return ' + txt.format(pourc(t_fft_return, t_fft_cpp)))
        print('return i' + txt.format(pourc(t_ifft_return, t_ifft_cpp)))
        # print('generic ' + txt.format(pourc(t_fft_generic, t_fft_cpp)))
        # print('generic i' + txt.format(pourc(t_ifft_generic, t_ifft_cpp)))


if __name__ == '__main__':

    n = 96

    def run(FFT):
        if FFT is None:
            return
        o = FFT(n, n, n)
        o.run_tests()
        o.run_benchs()
        compare_benchs(o, nb_time_execute=10)

    if nb_proc == 1 and rank == 0:
        for FFT in get_classes_seq().values():
            run(FFT)

    if nb_proc > 1:
        for FFT in get_classes_mpi().values():
            run(FFT)