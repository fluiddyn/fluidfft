
from glob import glob
import json
from copy import copy

import pandas as pd
import matplotlib.pyplot as plt

key_solver = 'ns2d'
hostname = 'cl7'
hostname = None

dicts = []

for path in glob('results_bench/result_bench2d*.json'):
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

times_names = {}
names = list(set(df['name']))

nb_procs = list(set(df['nb_proc']))
nb_procs.sort()
keys_times = ['t_fft_cpp', 't_ifft_cpp', 't_fft_as_arg', 't_ifft_as_arg']

for name in names:
    df_name = df[df['name'] == name]
    times = pd.DataFrame(index=nb_procs, columns=keys_times)

    for nb_proc in nb_procs:
        tmp = []
        for k in keys_times:
            tmp.append(df_name[df_name['nb_proc'] == nb_proc][k].min())
        times.loc[nb_proc] = tmp

    times_names[name] = times


plt.figure()
ax = plt.subplot()

for name, times in times_names.items():
    for k in keys_times:
        ts = times[k]
        ax.plot(ts, 'x')


# def plot_bench(nb_proc0=1):
#     df0 = df.loc[df['nb_proc'] == nb_proc0]
#     t_elapsed0 = df0['t_elapsed'].mean()
#     times = df['t_elapsed']

#     plt.figure()
#     ax = plt.subplot()
#     ax.plot(df['nb_proc'], nb_proc0*t_elapsed0/times, 'xr')
#     tmp = [nb_proc0, df['nb_proc'].max()]
#     ax.plot(tmp, tmp, 'b-')
#     ax.set_title('speed up')


# plot_bench(nb_proc0=1)
# plot_bench(nb_proc0=2)


plt.show()
