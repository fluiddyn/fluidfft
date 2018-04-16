
import unittest
import sys
import getpass

import matplotlib as mpl

mpl.use("Agg")

from fluiddyn.io import stdout_redirected

from fluiddyn.util import mpi

from .bench import bench_all, run
from . import bench_analysis


path_tmp = "/tmp/fluidfft_test_bench" + getpass.getuser()


class TestsBench(unittest.TestCase):

    def test2d(self):
        with stdout_redirected():
            args = "fluidfft-bench 24 24 -o".split()
            args.append(path_tmp)
            sys.argv = args
            run()
            if mpi.nb_proc > 1 and mpi.rank == 0:
                args = "fluidfft-bench-analysis 24 -d 2 -i".split()
                args.append(path_tmp)
                sys.argv = args
                bench_analysis.run()

    def test3d(self):
        with stdout_redirected():
            bench_all(
                dim="3d",
                n0=8,
                n1=None,
                n2=None,
                path_dir=path_tmp,
                skip_patterns=["p3dfft"],
            )


if __name__ == "__main__":
    unittest.main()
