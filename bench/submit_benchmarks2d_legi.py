#!/usr/bin/env python

from fluiddyn.clusters.legi import Calcul8 as Cluster

cluster = Cluster()
cluster.commands_setting_env = [
    'source /etc/profile',
    'export PATH="/home/users/augier3pi/opt/miniconda3/bin:$PATH"']


def submit(nb_nodes, nb_cores_per_node=None):
    if nb_cores_per_node is None:
        nb_cores_per_node = cluster.nb_cores_per_node
    nb_mpi = nb_cores_per_node*nb_nodes
    cluster.submit_command(
        'fluidfft-bench 2160 2160 '
        '-o /.fsnet/data/legi/calcul9/home/augier3pi/fluidfft_bench '
        '-n 20',
        name_run='fluidfft-bench_{:02d}'.format(nb_mpi),
        nb_nodes=nb_nodes,
        # nb_cores_per_node=nb_cores_per_node,
        nb_cores_per_node=cluster.nb_cores_per_node,
        walltime='00:40:00',
        nb_mpi_processes=nb_mpi, omp_num_threads=1,
        ask=False,
        delay_signal_walltime=None)


nb_nodes = 1
for nb_cores_per_node in [2, 4, 8, 10, 12, 16, 20]:
    if nb_cores_per_node > cluster.nb_cores_per_node:
        continue
    submit(nb_nodes, nb_cores_per_node)

for nb_nodes in [2]:
    submit(nb_nodes)
