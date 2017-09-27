"""

python benchs2d.py
mpirun -np 2 python benchs2d.py


"""


from __future__ import print_function, division

import os
import json
import socket

try:
    from time import perf_counter as time
except ImportError:
    # python 2.7
    from time import time

import numpy as np

from fluiddyn.util import mpi, time_as_str

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

    results_cpp = o.run_benchs(nb_time_execute)

    t_fft_as_arg, t_ifft_as_arg = bench_like_cpp_as_arg(o, nb_time_execute)
    t_fft_return, t_ifft_return = bench_like_cpp_return(o, nb_time_execute)

    if rank != 0:
        return

    # results_cpp is a tuple of size 2 only if rank == 0
    t_fft_cpp, t_ifft_cpp = results_cpp
    txt = 'fft is {:4.2f} % slower than cpp'
    print('as_arg ' + txt.format(pourc(t_fft_as_arg, t_fft_cpp)))
    print('as_arg i' + txt.format(pourc(t_ifft_as_arg, t_ifft_cpp)))
    print('return ' + txt.format(pourc(t_fft_return, t_fft_cpp)))
    print('return i' + txt.format(pourc(t_ifft_return, t_ifft_cpp)))

    results = {
        'name': o.get_short_name(),
        't_fft_cpp': t_fft_cpp,
        't_ifft_cpp': t_ifft_cpp,
        't_fft_as_arg': t_fft_as_arg,
        't_ifft_as_arg': t_ifft_as_arg,
        't_fft_return': t_fft_return,
        't_ifft_return': t_ifft_return}

    return results


def bench_all(n0=1024*2, n1=None):

    if n1 is None:
        n1 = n0

    def run(FFT2D):
        if FFT2D is None:
            return
        o = FFT2D(n0, n1)
        o.run_tests()
        o.run_benchs()
        return compare_benchs(o, nb_time_execute=10)

    t_as_str = time_as_str()

    if nb_proc == 1:
        classes = get_classes_seq()
    else:
        classes = get_classes_mpi()

    classes = {k: cls for k, cls in classes.items() if cls is not None}

    results_classes = []
    for key, FFT2D in sorted(classes.items()):
        results_classes.append(run(FFT2D))

    if rank > 0:
        return

    path_results = 'results_bench'
    if not os.path.exists(path_results):
        os.mkdir(path_results)

    pid = os.getpid()
    nfile = (
        'result_bench2d_{}x{}'.format(n0, n1) +
        '_' + t_as_str + '_{}'.format(pid) + '.json')

    path = os.path.join(path_results, nfile)

    results = {
        'n0': n0,
        'n1': n1,
        'nb_proc': mpi.nb_proc,
        'pid': pid,
        'time_as_str': t_as_str,
        'hostname': socket.gethostname(),
        'benchmarks_classes': results_classes}

    with open(path, 'w') as f:
        json.dump(results, f, sort_keys=True)
        f.write('\n')

if __name__ == '__main__':
    bench_all(1024//2, 1024//2)
