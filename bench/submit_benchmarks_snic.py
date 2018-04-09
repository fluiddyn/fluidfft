#!/usr/bin/env python
import os
import numpy as np
from fluiddyn.clusters.snic import ClusterSNIC as Cluster


# Parameters
# ----------
## n0 = 2 ** 10; 'Triolith / Beskow'
n = (384, 1152, 1152); nb_cores = [2, 4, 8, 16, 32]; nodes = [1, 2, 4, 8, 16, 32, 64, 128]

## n0 = 2**6 * 3**2 * 7; 'Kebnekaise'
# n = (1008,); nb_cores = [2, 4, 8, 12, 16, 21, 24, 28]; nodes = [2, 3, 4, 6]


def shape(join_with=' '):
    n_as_str = [str(i) for i in n]
    return join_with.join(n_as_str)


# argv = dict(dim='2d', nh=f'{shape()} -d 2', time='00:04:00')  # 2D benchmarks
argv = dict(dim='3d', nh=f'{shape()} -d 3', time='00:20:00')  # 3D benchmarks
# mode = 'intra'
mode = 'inter'
# mode = 'inter-intra'


def init_cluster():
    global output_dir

    cluster = Cluster()
    output_dir = os.path.abspath(
        f"./../../fluidfft-bench-results/"
        f"{cluster.name_cluster}_{shape('x')}")

    cluster.max_walltime = '00:20:01'
    if cluster.name_cluster == 'beskow':
        cluster.default_project = '2017-12-20'
        interactive=False
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
    cluster.submit_command(
        f"fluidfft-bench {argv['nh']} -o {output_dir} -n 20",
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
