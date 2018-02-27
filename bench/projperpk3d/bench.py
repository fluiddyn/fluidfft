
import numpy as np

from proj_native import proj as proj_native
from proj_default import proj as proj_default
from proj1_default import proj as proj1_default
from proj2_default import proj as proj2_default
from proj_omp import proj as proj_omp

from proj import proj as proj_py

# from fluidfft.fft3d.util_pythran import project_perpk3d as proj_fft

n0 = n1 = n2 = 128

shape = (n0, n1, n2//2+1)

arr_c = 2.3j + np.ones(shape, dtype=np.complex128)

arr = np.ones(shape)
