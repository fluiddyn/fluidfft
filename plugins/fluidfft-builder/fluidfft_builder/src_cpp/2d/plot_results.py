from runpy import run_path

import numpy as np

import matplotlib.pyplot as plt

array = np.array

d = run_path("Results_bench/bench.py")

nb_procs = d["nb_procs"]
times_fft = d["times_fft"]
times_ifft = d["times_ifft"]

fig = plt.figure()
ax = plt.gca()

for cls, nb_proc in nb_procs.items():
    ax.plot(nb_proc, times_fft[cls], "bo--")
    ax.plot(nb_proc, times_ifft[cls], "ro--")

ax.set_xlabel("number of processes")
ax.set_ylabel("time (s)")

ind_base = 0
cls_base = "FFT2DMPIWithFFTW1D"

nb_proc_base = nb_procs[cls_base][ind_base]


fig = plt.figure()
ax = plt.gca()

for cls, nb_proc in nb_procs.items():

    speedup_fft = times_fft[cls_base][ind_base] / times_fft[cls]
    speedup_ifft = times_ifft[cls_base][ind_base] / times_ifft[cls]

    ax.plot(nb_proc, nb_proc_base * speedup_fft / nb_proc, "bo--")
    ax.plot(nb_proc, nb_proc_base * speedup_ifft / nb_proc, "ro--")

ax.set_xlabel("number of processes")
ax.set_ylabel("speedup divided by the number of processes")


plt.show()
