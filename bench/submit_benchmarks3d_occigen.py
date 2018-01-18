
from fluiddyn.clusters.cines import Occigen as Cluster

cluster = Cluster()

for nb_nodes in [1, 2, 3, 4, 6, 8, 12, 16, 20]:
    cluster.submit_command(
        'fluidfft-bench 1152 1152 384 -d 3 -o ~/fluidfft_bench -n 20',
        name_run=f'fluidfft-bench1152x1152x384_{nb_nodes}',
        nb_nodes=nb_nodes,
        walltime='00:30:00',
        ask=False)
