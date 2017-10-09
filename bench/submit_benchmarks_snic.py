#!/usr/bin/env python
import os
import numpy as np
from fluiddyn.clusters.snic import ClusterSNIC as Cluster
import fluidfft


argv = ('2D', '1024 -d 2')  # 2D benchmarks
# argv = ('3D', '960 960 240')  # 3D benchmarks
# mode = 'intra'
mode = 'inter'


def init_cluster():
    global output_dir

    cluster = Cluster()
    if cluster.name_cluster == 'beskow':
        cluster.default_project = '2016-34-10'
        interactive=False
    else:
        cluster.default_project = 'SNIC2016-34-10'
        interactive=True

    output_dir = os.path.abspath('{}/../doc/benchmarks/snic_{}_{}'.format(
        os.path.split(fluidfft.__file__)[0], cluster.name_cluster, argv[0]))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print('Output directory: ', output_dir)
    cluster.commands_unsetting_env.extend([
        'fluidinfo -o ' + output_dir])
    return cluster, interactive


def submit(cluster, interactive, nb_nodes, nb_cores_per_node=None):
    if nb_cores_per_node is None:
        nb_cores_per_node = cluster.nb_cores_per_node

    nb_mpi = nb_cores_per_node * nb_nodes
    cluster.submit_command(
        'fluidfft-bench ' + argv[1] + ' -o ' + output_dir,
        name_run='bench_{}_{}'.format(argv[0], nb_mpi),
        nb_nodes=nb_nodes,
        nb_cores_per_node=nb_cores_per_node,
        walltime='00:01:00',
        nb_mpi_processes=nb_mpi, omp_num_threads=1,
        ask=False, bash=False, interactive=interactive)


cluster, interactive = init_cluster()
if mode == 'intra':
    nb_nodes = 1
    # nb_cores = [2, 4, 8, 12, 16, 20, 24, 28, 32]
    nb_cores = 4 * np.arange(0, 9)
    nb_cores[0] = 2

    for nb_cores_per_node in nb_cores:
        if nb_cores_per_node > cluster.nb_cores_per_node:
            continue
        submit(cluster, interactive, nb_nodes, nb_cores_per_node)
else:
    nodes = [2, 4, 8]
    for nb_nodes in nodes:
        submit(cluster, interactive, nb_nodes)
