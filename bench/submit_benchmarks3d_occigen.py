#!/usr/bin/env python

from fluiddyn.clusters.cines import Occigen as Cluster

cluster = Cluster()

# n0 = 384
# n1 = n2 = 1152

n0 = n1 = n2 = 1152

for nb_nodes in [4, 6, 8, 16, 32, 64, 128, 256, 420]:
    cluster.submit_command(
        f'fluidfft-bench {n0} {n1} {n2} -d 3 -o ~/fluidfft_bench -n 10',
        name_run=f'fluidfft-bench{n0}x{n1}x{n2}_{nb_nodes:02d}',
        nb_nodes=nb_nodes,
        walltime='00:30:00',
        ask=False)
