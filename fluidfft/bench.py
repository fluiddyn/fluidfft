"""

python benchs2d.py
mpirun -np 2 python benchs2d.py


"""

from __future__ import print_function, division

import os
import json
import socket
import argparse

from time import time
# try:
#     from time import perf_counter as clock
# except ImportError:
#     # python 2.7
#     from time import clock as time

import numpy as np

from . import __version__

from fluiddyn.util import mpi, time_as_str

path_results = '/tmp/fluidfft_bench'

rank = mpi.rank
nb_proc = mpi.nb_proc


print_old = print


def print(*args, **kwargs):
    if mpi.rank == 0:
        print_old(*args, **kwargs)


def bench_like_cpp_as_arg(o, nb_time_execute=20):

    fieldX = np.ones(o.get_shapeX_loc(), dtype=float, order='C')
    fieldK = np.empty(o.get_shapeK_loc(), dtype=np.complex128, order='C')

    times = np.empty([nb_time_execute])
    for i in range(nb_time_execute):
        t_start = time()
        o.fft_as_arg(fieldX, fieldK)
        t_end = time()
        times[i] = t_end - t_start
    t_fft = np.median(times)
    print('time fft_as_arg:  {}'.format(t_fft))

    for i in range(nb_time_execute):
        t_start = time()
        o.ifft_as_arg(fieldK, fieldX)
        t_end = time()
        times[i] = t_end - t_start
    t_ifft = np.median(times)
    print('time ifft_as_arg: {}'.format(t_ifft))
    return t_fft, t_ifft


def bench_like_cpp_return(o, nb_time_execute=20):

    fieldX = np.ones(o.get_shapeX_loc(), dtype=float, order='C')
    fieldK = np.empty(o.get_shapeK_loc(), dtype=np.complex128, order='C')

    times = np.empty([nb_time_execute])
    for i in range(nb_time_execute):
        t_start = time()
        o.fft(fieldX)
        t_end = time()
        times[i] = t_end - t_start
    t_fft = np.median(times)
    print('time return_fft:  {}'.format(t_fft))

    for i in range(nb_time_execute):
        t_start = time()
        o.ifft(fieldK)
        t_end = time()
        times[i] = t_end - t_start
    t_ifft = np.median(times)
    print('time return_ifft: {}'.format(t_ifft))
    return t_fft, t_ifft


def pourc(t_slow, t_fast):
    return 100. * (t_slow - t_fast) / t_fast


def compare_benchs(o, nb_time_execute=20):

    t_start = time()
    results_cpp = o.run_benchs(nb_time_execute)
    t_fft_as_arg, t_ifft_as_arg = bench_like_cpp_as_arg(o, nb_time_execute)
    t_fft_return, t_ifft_return = bench_like_cpp_return(o, nb_time_execute)
    t_end = time()

    if rank != 0:
        return

    # results_cpp is a tuple of size 2 only if rank == 0
    t_fft_cpp, t_ifft_cpp = results_cpp
    txt = 'fft is {:4.2f} % slower than cpp'
    print('as_arg ' + txt.format(pourc(t_fft_as_arg, t_fft_cpp)))
    print('as_arg i' + txt.format(pourc(t_ifft_as_arg, t_ifft_cpp)))
    print('return ' + txt.format(pourc(t_fft_return, t_fft_cpp)))
    print('return i' + txt.format(pourc(t_ifft_return, t_ifft_cpp)))

    name = o.get_short_name()
    print('total time bench for lib ' + name + ': {:4.2f} s'.format(
        t_end-t_start))

    results = {
        'name': name,
        't_fft_cpp': t_fft_cpp,
        't_ifft_cpp': t_ifft_cpp,
        't_fft_as_arg': t_fft_as_arg,
        't_ifft_as_arg': t_ifft_as_arg,
        't_fft_return': t_fft_return,
        't_ifft_return': t_ifft_return}

    return results


def bench_all(dim='2d', n0=1024*2, n1=None, n2=None, path_dir=path_results,
              skip_patterns=None):

    if n1 is None:
        n1 = n0

    if n2 is None:
        n2 = n0

    if dim == '2d':
        from fluidfft.fft2d import get_classes_seq, get_classes_mpi
        str_grid = '{}x{}'.format(n0, n1)
    elif dim == '3d':
        from fluidfft.fft3d import get_classes_seq, get_classes_mpi
        str_grid = '{}x{}x{}'.format(n0, n1, n2)
    else:
        raise ValueError("dim has to be in ['2d', '3d']")

    def run(FFT):
        if FFT is None:
            return
        if 'fft3d' in FFT.__name__.lower():
            o = FFT(n0, n1, n2)
        else:
            o = FFT(n0, n1)
        o.run_tests()
        # o.run_benchs()
        return compare_benchs(o, nb_time_execute=50)

    t_as_str = time_as_str()

    if nb_proc == 1:
        classes = get_classes_seq()
    else:
        classes = get_classes_mpi()

    classes = {k: cls for k, cls in classes.items() if cls is not None}

    if skip_patterns is not None:
        for pattern in skip_patterns:
            classes = {k: cls for k, cls in classes.items()
                       if pattern not in k}

    results_classes = []
    for key, FFT in sorted(classes.items()):
        results_classes.append(run(FFT))

    if rank > 0:
        return

    if not os.path.exists(path_dir):
        os.makedirs(path_dir)

    pid = os.getpid()
    nfile = (
        'result_bench' + dim + '_' + str_grid +
        '_{}'.format(mpi.nb_proc) + '_' + t_as_str +
        '_{}'.format(pid) + '.json')

    path = os.path.join(path_dir, nfile)

    results = {
        'n0': n0,
        'n1': n1,
        'nb_proc': mpi.nb_proc,
        'pid': pid,
        'time_as_str': t_as_str,
        'hostname': socket.gethostname(),
        'benchmarks_classes': results_classes}

    if dim == '3d':
        results['n2'] = n2

    with open(path, 'w') as f:
        json.dump(results, f, sort_keys=True)
        f.write('\n')

    print('results benchmarks saved in\n' + path)


class MyValueError(ValueError):
    pass


def parse_args_dim(parser):

    args = parser.parse_args()

    dim = args.dim
    n0 = args.n0
    n1 = args.n1
    n2 = args.n2

    if dim is None:
        if n0 is not None and n1 is not None and n2 is None:
            dim = '2d'
        elif n0 is not None and n1 is not None and n2 is not None:
            dim = '3d'
        else:
            print(
                'Cannot determine which shape you want to use for this bench '
                "('2d' or '3d')")
            raise MyValueError

    if dim.lower() in ['3', '3d']:
        if n2 is None:
            n2 = n0
        dim = '3d'
    elif dim.lower() in ['2', '2d']:
        dim = '2d'
    else:
        raise ValueError('dim should not be {}'.format(dim))

    if n1 is None:
        n1 = n0

    args.dim = dim
    args.n0 = n0
    args.n1 = n1
    args.n2 = n2

    return args


def run():
    parser = argparse.ArgumentParser(
        prog='fluidfft-bench',
        description='Perform benchmarks of fluidfft classes.')

    parser.add_argument('n0', nargs='?', type=int, default=64)
    parser.add_argument('n1', nargs='?', type=int, default=None)
    parser.add_argument('n2', nargs='?', type=int, default=None)

    parser.add_argument('-V', '--version',
                        action='version',
                        version=__version__)

    parser.add_argument('-d', '--dim', default=None)

    parser.add_argument('-o', '--output_dir', default=path_results)

    try:
        args = parse_args_dim(parser)
    except MyValueError:
        return

    if args.dim == '3d':
        bench_all(args.dim, args.n0, args.n1, args.n2,
                  path_dir=args.output_dir)
    else:
        bench_all(args.dim, args.n0, args.n1, path_dir=args.output_dir)
