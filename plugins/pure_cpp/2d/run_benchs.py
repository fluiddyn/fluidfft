import os

import subprocess32 as subprocess

import numpy as np

output_dir = "Results_bench"


def call_bash(commands):
    print(commands)
    return subprocess.check_output(commands, shell=True)


N0 = N1 = 256 * 2 * 2

nb_procs_all = np.array([1, 2, 4, 8])

nb_procs = {}
times_fft = {}
times_ifft = {}

classes = []

for i, nb_proc in enumerate(nb_procs_all):

    command = "./test_bench.out --N0={} --N1={}".format(N0, N1)

    if nb_proc != 1:
        command = "mpirun -np {} ".format(nb_proc) + command

    txt = call_bash(command)

    blocks = txt.split("--------")
    # print('\n'.join(lines))

    for block in blocks:
        if "Initialization" in block:
            lines = block.splitlines()
            for line in lines:
                if line.startswith("Initialization ("):
                    cls = line.split("(")[1].split(")")[0]
                    if cls not in classes:
                        classes.append(cls)
                        nb_procs[cls] = []
                        times_fft[cls] = []
                        times_ifft[cls] = []

            for line in lines:
                if line.startswith("nb_proc:"):
                    nb_procs[cls].append(int(line.split()[1]))
                if line.startswith("time fft"):
                    times_fft[cls].append(float(line.split()[3]))
                if line.startswith("time ifft"):
                    times_ifft[cls].append(float(line.split()[3]))

            print(
                "class "
                + cls
                + ": times fft and ifft: {:5.3f} ; {:5.3f}".format(
                    times_fft[cls][-1], times_ifft[cls][-1]
                )
            )

src = """
from numpy import array

nb_procs = {}
times_fft = {}
times_ifft = {}
"""

for cls in classes:
    src += (
        f"\nnb_procs['{cls}'] = "
        + repr(np.array(nb_procs[cls]))
        + f"\ntimes_fft['{cls}'] = "
        + repr(np.array(times_fft[cls]))
        + "\ntimes_ifft['{cls}'] = "
        + repr(np.array(times_ifft[cls]))
    )

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

with open(output_dir + "/bench.py", "w", encoding="utf8") as file:
    file.write(src + "\n")
