
import numpy as np

# from proj_native import proj as proj_native
# from proj_default import proj as proj_default
# try:
#     from proj_omp import proj as proj_omp
# except ImportError:
#     pass

from proj_native import proj_loop as proj_native
from proj_default import proj_loop as proj_default
try:
    from proj_omp import proj_loop as proj_omp
except ImportError:
    pass


from proj import proj as proj_py

from fluidfft.fft3d.util_pythran import project_perpk3d as proj_fft

n0 = n1 = n2 = 128

shape = (n0, n1, n2//2+1)

c0 = 1.3j + np.ones(shape, dtype=np.complex128)
c1 = 2.3j + np.ones(shape, dtype=np.complex128)
c2 = 3.3j + np.ones(shape, dtype=np.complex128)

a0 = np.ones(shape)
a1 = np.ones(shape)
a2 = np.ones(shape)
a3 = np.ones(shape)
