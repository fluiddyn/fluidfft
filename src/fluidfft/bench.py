"""Benchmarking of fluidfft classes (:mod:`fluidfft.bench`)
===========================================================


"""

import os
import json
import socket
import argparse
import gc
import sys

from time import time

# try:
#     from time import perf_counter as clock
# except ImportError:
#     # python 2.7
#     from time import clock as time

import numpy as np

from . import __version__

from fluiddyn.util import mpi, time_as_str

path_results = "/tmp/fluidfft_bench"

rank = mpi.rank
nb_proc = mpi.nb_proc

print_old = print
print = mpi.printby0


def _format_times(times):
    return (
        "\tmedian = {:.6f} s\n" "\tmean =   {:.6f} s\n" "\tmin =    {:.6f} s"
    ).format(np.median(times), times.mean(), times.min())


def bench_like_cpp_as_arg(obj, nb_exec=20):
    fieldX = obj.create_arrayX(1)
    fieldK = obj.create_arrayK(1)

    times = np.empty([nb_exec])
    gc.disable()
    for i in range(nb_exec):
        t_start = time()
        result = obj.fft_as_arg(fieldX, fieldK)
        t_end = time()
        del result
        times[i] = t_end - t_start
    t_fft = np.median(times)
    print("time fft_as_arg:\n" + _format_times(times))

    for i in range(nb_exec):
        t_start = time()
        result = obj.ifft_as_arg(fieldK, fieldX)
        t_end = time()
        del result
        times[i] = t_end - t_start
    gc.enable()
    t_ifft = np.median(times)
    print("time ifft_as_arg:\n" + _format_times(times))
    return t_fft, t_ifft


def bench_like_cpp_return(obj, nb_exec=20):
    fieldX = obj.create_arrayX(1)
    fieldK = obj.create_arrayK(1)

    times = np.empty([nb_exec])
    gc.disable()
    for i in range(nb_exec):
        t_start = time()
        result = obj.fft(fieldX)
        t_end = time()
        del result
        times[i] = t_end - t_start
    t_fft = np.median(times)
    print("time return_fft:\n" + _format_times(times))

    for i in range(nb_exec):
        t_start = time()
        result = obj.ifft(fieldK)
        t_end = time()
        del result
        times[i] = t_end - t_start
    gc.enable()
    t_ifft = np.median(times)
    print("time return_ifft:\n" + _format_times(times))
    return t_fft, t_ifft


def pourc(t_slow, t_fast):
    return 100.0 * (t_slow - t_fast) / t_fast


def compare_benchs(o, nb_exec=20):
    t_start = time()
    o.run_benchs(1)
    results_cpp = o.run_benchs(nb_exec)
    t_fft_as_arg, t_ifft_as_arg = bench_like_cpp_as_arg(o, nb_exec)
    t_fft_return, t_ifft_return = bench_like_cpp_return(o, nb_exec)
    t_end = time()

    if rank != 0:
        return

    # results_cpp is a tuple of size 2 only if rank == 0
    t_fft_cpp, t_ifft_cpp = results_cpp
    txt = "fft is {:4.2f} % slower than cpp"
    print("as_arg " + txt.format(pourc(t_fft_as_arg, t_fft_cpp)))
    print("as_arg i" + txt.format(pourc(t_ifft_as_arg, t_ifft_cpp)))
    print("return " + txt.format(pourc(t_fft_return, t_fft_cpp)))
    print("return i" + txt.format(pourc(t_ifft_return, t_ifft_cpp)))

    name = o.get_short_name()
    print(
        "total time bench for lib " + name + ": {:4.2f} s".format(t_end - t_start)
    )

    # may be necessary to force print order (Python, C++)
    sys.stdout.flush()

    results = {
        "name": name,
        "t_fft_cpp": t_fft_cpp,
        "t_ifft_cpp": t_ifft_cpp,
        "t_fft_as_arg": t_fft_as_arg,
        "t_ifft_as_arg": t_ifft_as_arg,
        "t_fft_return": t_fft_return,
        "t_ifft_return": t_ifft_return,
    }

    return results


def bench_all(
    dim="2d",
    n0=1024 * 2,
    n1=None,
    n2=None,
    path_dir=path_results,
    # tmp: to skip failing test
    skip_patterns=("dask",),
    nb_exec=None,
):
    if n1 is None:
        n1 = n0

    if n2 is None:
        n2 = n0

    if dim == "2d":
        from fluidfft.fft2d import get_classes_seq, get_classes_mpi

        str_grid = "{}x{}".format(n0, n1)
    elif dim == "3d":
        from fluidfft.fft3d import get_classes_seq, get_classes_mpi

        str_grid = "{}x{}x{}".format(n0, n1, n2)
    else:
        raise ValueError("dim has to be in ['2d', '3d']")

    if nb_exec is None:
        if dim == "2d":
            nb_exec = 50
        else:
            nb_exec = 20

    def run(FFT):
        cls_name = FFT.__name__.lower()
        str_start = "Starting benchmark for class " + FFT.__name__
        print("\n" + str_start + "\n" + len(str_start) * "=")
        sys.stdout.flush()
        if FFT is None:
            return

        try:
            if "fft3d" in cls_name:
                obj = FFT(n0, n1, n2)
            else:
                obj = FFT(n0, n1)
        except ValueError:
            print("ValueError during initialization for class " + cls_name)
            sys.stdout.flush()
            return

        obj.run_tests()
        return compare_benchs(obj, nb_exec=nb_exec)

    t_as_str = time_as_str()

    if nb_proc == 1:
        classes = get_classes_seq()
    else:
        classes = get_classes_mpi()

    classes = {k: cls for k, cls in classes.items() if cls is not None}

    if skip_patterns is not None:
        for pattern in skip_patterns:
            classes = {k: cls for k, cls in classes.items() if pattern not in k}

    results_classes = []
    for key, FFT in sorted(classes.items()):
        result = run(FFT)
        if result is not None:
            results_classes.append(result)

    if rank > 0:
        return

    if not os.path.exists(path_dir):
        os.makedirs(path_dir)

    pid = os.getpid()
    nfile = (
        "result_bench"
        + dim
        + "_"
        + str_grid
        + "_{:04d}".format(mpi.nb_proc)
        + "_"
        + t_as_str
        + "_{}".format(pid)
        + ".json"
    )

    path = os.path.join(path_dir, nfile)

    results = {
        "n0": n0,
        "n1": n1,
        "nb_proc": mpi.nb_proc,
        "pid": pid,
        "time_as_str": t_as_str,
        "hostname": socket.gethostname(),
        "benchmarks_classes": results_classes,
    }

    if dim == "3d":
        results["n2"] = n2

    with open(path, "w") as f:
        json.dump(results, f, indent=1, sort_keys=True)
        f.write("\n")

    print("results benchmarks saved in\n" + path)


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
            dim = "2d"
        elif n0 is not None and n1 is not None and n2 is not None:
            dim = "3d"
        else:
            print(
                "Cannot determine which shape you want to use for this bench "
                "('2d' or '3d')"
            )
            raise MyValueError

    if dim.lower() in ["3", "3d"]:
        if n2 is None:
            n2 = n0
        dim = "3d"
    elif dim.lower() in ["2", "2d"]:
        dim = "2d"
    else:
        raise ValueError("dim should not be {}".format(dim))

    if n1 is None:
        n1 = n0

    args.dim = dim
    args.n0 = n0
    args.n1 = n1
    args.n2 = n2

    return args


def run():
    parser = argparse.ArgumentParser(
        prog="fluidfft-bench",
        description="Perform benchmarks of fluidfft classes.",
    )

    parser.add_argument("n0", nargs="?", type=int, default=64)
    parser.add_argument("n1", nargs="?", type=int, default=None)
    parser.add_argument("n2", nargs="?", type=int, default=None)

    parser.add_argument("-V", "--version", action="version", version=__version__)

    parser.add_argument("-d", "--dim", default=None)
    parser.add_argument("-o", "--output_dir", default=path_results)
    parser.add_argument("-n", "--nb_exec", type=int, default=None)

    try:
        args = parse_args_dim(parser)
    except MyValueError:
        return

    bench_all(
        args.dim,
        args.n0,
        args.n1,
        args.n2,
        path_dir=args.output_dir,
        nb_exec=args.nb_exec,
    )
