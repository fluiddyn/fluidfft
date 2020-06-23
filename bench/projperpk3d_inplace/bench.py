import numpy as np

try:
    from proj_native import proj as proj_native
    from proj_native import proj_loop as proj_loop_native
except ImportError:
    pass

try:
    from proj_default import proj as proj_default
    from proj_default import proj_loop as proj_loop_default
except ImportError:
    pass

try:
    from proj_omp import proj as proj_omp
    from proj_omp import proj_loop as proj_loop_omp
except ImportError:
    pass


from proj import proj as proj_py

# from fluidfft.fft3d.util_pythran import project_perpk3d as proj_fft
# from fluidfft.fft3d.operators import OperatorsPseudoSpectral3D

n0 = n1 = n2 = 128
# lx = 2*np.pi

# oper = OperatorsPseudoSpectral3D(n2, n1, n0, lx, lx, lx)
# proj_fft =

shape = (n0, n1, n2 // 2 + 1)

c0 = 1.3j + np.ones(shape, dtype=np.complex128)
c1 = 2.3j + np.ones(shape, dtype=np.complex128)
c2 = 3.3j + np.ones(shape, dtype=np.complex128)

a0 = np.ones(shape)
a1 = np.ones(shape)
a2 = np.ones(shape)
a3 = np.ones(shape)
