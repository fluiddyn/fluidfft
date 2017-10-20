
import unittest

from fluiddyn.io import stdout_redirected

from fluiddyn.util import mpi

from .bench import bench_all
from .bench_analysis import plot_scaling

path_tmp = '/tmp/fluidfft_test_bench'


class TestsBench(unittest.TestCase):

    def test2d(self):
        n0 = 24
        with stdout_redirected():
            bench_all(dim='2d', n0=24, n1=None, n2=None, path_dir=path_tmp)
            if mpi.nb_proc > 1 and mpi.rank == 0:
                plot_scaling(path_tmp, None, n0, n0, '2d', show=False)

    def test3d(self):
        with stdout_redirected():
            bench_all(dim='3d', n0=8, n1=None, n2=None, path_dir=path_tmp,
                      skip_patterns=['p3dfft'])

if __name__ == '__main__':
    unittest.main()
