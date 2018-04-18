#!/usr/bin/env python
import os
import numpy as np
# from fluiddyn.clusters.snic import ClusterSNIC as Cluster
from fluiddyn.clusters.snic import Beskow36 as Cluster


# Parameters
# ----------
## 'Triolith / Beskow'
# n = (384, 1152, 1152); nb_cores = [2, 4, 8, 16, 24]; nodes = [1, 2, 4, 16, 20, 40, 50]  # Similar to Occigen
# n = (384, 1152, 1152); nb_cores = [2, 4, 8, 16, 32];  nodes = [1, 2, 4, 6, 8, 9, 12, 16, 18, 32, 36]
# n = (1152, 1152, 1152); nb_cores = [2, 4, 8, 16, 32]; nodes = [1, 2, 4, 6, 8, 9, 12, 16, 18, 32, 36, 64]  # , 128, 256, 384, 512]
n = (1152, 1152, 1152); nb_cores = [2, 4, 8, 16, 32]; nodes = [96, 128, 192, 256, 384, 512]

## 'Kebnekaise'
# n = (1008,); nb_cores = [2, 4, 8, 12, 16, 21, 24, 28]; nodes = [2, 3, 4, 6]


def shape(join_with=' '):
    n_as_str = [str(i) for i in n]
    return join_with.join(n_as_str)


# argv = dict(dim='2d', nh=f'{shape()} -d 2', time='00:04:00')  # 2D benchmarks
argv = dict(dim='3d', nh=f'{shape()} -d 3', time='00:20:00')  # 3D benchmarks small
# argv = dict(dim='3d', nh=f'{shape()} -d 3', time='01:00:00')  # 3D benchmarks large
# mode = 'intra'
mode = 'inter'
# mode = 'inter-intra'
dry_run = False
# dry_run = True


def init_cluster():
    global output_dir

    cluster = Cluster()
    output_dir = os.path.abspath(
        f"./../../fluidfft-bench-results/"
        f"{cluster.name_cluster}_{shape('x')}")

    cluster.max_walltime = argv['time']
    if cluster.name_cluster == 'beskow':
        cluster.default_project = '2017-12-20'
        interactive = True
        cluster.nb_cores_per_node = 32  # TODO: Change this if other shapes are used
        cluster.cmd_run_interactive = f'aprun -N {cluster.nb_cores_per_node}'
        cluster.commands_unsetting_env.insert(
            0, f'aprun -n 1 fluidinfo -o {output_dir}')
    else:
        cluster.default_project = 'SNIC2017-12-20'
        interactive=True
        cluster.commands_unsetting_env.insert(
            0, f'fluidinfo -o {output_dir}')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print('Output directory: ', output_dir)
    return cluster, interactive


def submit(cluster, interactive, nb_nodes, nb_cores_per_node=None):
    if nb_cores_per_node is None:
        nb_cores_per_node = cluster.nb_cores_per_node

    nb_mpi = nb_cores_per_node * nb_nodes
    cmd = f"fluidfft-bench {argv['nh']} -o {output_dir} -n 20"
    if dry_run:
        print(f'np={nb_mpi} N={nb_nodes}', cmd)
    else:
        cluster.submit_command(
            cmd,
            name_run=f"fft{argv['dim']}_{nb_mpi}",
            nb_nodes=nb_nodes,
            nb_cores_per_node=nb_cores_per_node,
            walltime=argv['time'],
            nb_mpi_processes=nb_mpi, omp_num_threads=1,
            ask=False, bash=False, interactive=interactive)


cluster, interactive = init_cluster()
if 'intra' in mode:
    nb_nodes = 1
    for nb_cores_per_node in nb_cores:
        if nb_cores_per_node > cluster.nb_cores_per_node:
            continue
        submit(cluster, interactive, nb_nodes, nb_cores_per_node)

if 'inter' in mode:
    for nb_nodes in nodes:
        submit(cluster, interactive, nb_nodes)
