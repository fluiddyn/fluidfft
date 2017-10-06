
from glob import glob
import json
from copy import copy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from .bench import (path_results, argparse, __version__, parse_args_dim,
                    MyValueError)


def load_bench(path_dir, hostname, dim):

    dicts = []
    for path in glob(path_dir + '/result_bench{}*.json'.format(dim)):
        with open(path) as f:
            d = json.load(f)
        if hostname is not None and not d['hostname'].startswith(hostname):
            continue

        d0 = {k: v for k, v in d.items() if k != 'benchmarks_classes'}

        ds = []
        for subd in d['benchmarks_classes']:
            tmp = copy(d0)
            for k, v in subd.items():
                tmp[k] = v
            ds.append(tmp)

        dicts.extend(ds)

    df = pd.DataFrame(dicts)
    df = df[df.columns.difference(['hostname', 'pid', 'time_as_str'])]
    return df


def filter_by_shape(df, n0, n1):
    df = df[(df.n0 == n0) & (df.n1 == n1)]
    return df[df.columns.difference(['n0', 'n1'])]


def plot_scaling(path_dir, hostname, n0, n1, dim, show=True):

    df = load_bench(path_dir, hostname, dim)
    df = filter_by_shape(df, n0, n1)

    # for "scaling" (mpi)
    df = df[df.nb_proc > 1]

    if df.empty:
        raise ValueError('No benchmarks corresponding to the input parameters')

    if show:
        print(df)

    nb_proc_min = df.nb_proc.min()

    df3 = df.groupby(['name', 'nb_proc']).quantile(q=0.2)

    keys_fft = [k for k in df3.columns if k.startswith('t_fft')]
    keys_ifft = [k for k in df3.columns if k.startswith('t_ifft')]

    df_fft = df3[keys_fft]
    df_ifft = df3[keys_ifft]

    df_fft_nb_proc_min = df_fft.xs(nb_proc_min, level=1)
    df_ifft_nb_proc_min = df_ifft.xs(nb_proc_min, level=1)

    def get_min(df):
        m = df.as_matrix()
        i0, i1 = np.unravel_index(np.argmin(m), m.shape)
        mymin = m[i0, i1]
        ind = df.index[i0]
        key = df.columns[i1]
        return mymin, ind, key

    t_min_fft, name_min_fft, key_min_fft = get_min(
        df_fft_nb_proc_min)
    t_min_ifft, name_min_ifft, key_min_ifft = get_min(
        df_ifft_nb_proc_min)

    fig = plt.figure(figsize=[15, 5])
    ax0 = plt.subplot(121)
    ax1 = plt.subplot(122)

    for name in df3.index.levels[0]:
        tmp = df3.loc[name]
        # print(name)

        for k in keys_fft:
            speedup = t_min_fft/tmp[k]*nb_proc_min
            ax0.plot(
                speedup.index, speedup.values, 'x-',
                label='{}, {}'.format(name, k))

        for k in keys_ifft:
            # print(k)
            speedup = t_min_ifft/tmp[k]*nb_proc_min
            ax1.plot(
                speedup.index, speedup.values, 'x-',
                label='{}, {}'.format(name, k))

        for ax in [ax0, ax1]:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            theoratical = [speedup.index.min(), speedup.index.max()]
            ax.plot(theoratical, theoratical, '-k')

    for ax in [ax0, ax1]:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('number of processes')
        ax.set_ylabel('speedup')

    ax0.set_title('Best for {} procs: {}, {} ({:.2f} ms)'.format(
        nb_proc_min, name_min_fft, key_min_fft, t_min_fft*1000))
    ax1.set_title('Best for {} procs: {}, {} ({:.2f} ms)'.format(
        nb_proc_min, name_min_ifft, key_min_ifft, t_min_ifft*1000))

    ax0.legend()
    ax1.legend()

    if show:
        plt.show()
    return fig


def run():
    parser = argparse.ArgumentParser(
        prog='fluidfft-bench-analysis',
        description='Plots on benchmarks of fluidfft classes.')

    parser.add_argument('n0', nargs='?', type=int, default=64)
    parser.add_argument('n1', nargs='?', type=int, default=None)
    parser.add_argument('n2', nargs='?', type=int, default=None)

    parser.add_argument('-V', '--version',
                        action='version',
                        version=__version__)

    parser.add_argument('-d', '--dim', default=None)
    parser.add_argument('-i', '--input_dir', default=path_results)
    parser.add_argument('--hostname', default=None)

    try:
        args = parse_args_dim(parser)
    except MyValueError:
        return

    if args.dim == '3d':
        raise NotImplementedError
    else:
        plot_scaling(args.input_dir, args.hostname, args.n0, args.n1, args.dim)


# dim = '2d'

# hostname = 'cl7'
# hostname = None

# n0 = 512
# n1 = 512

# type_plot = 'hist'
# type_plot = 'scaling'

# input_dir = ''


