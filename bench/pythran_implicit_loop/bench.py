
import numpy as np
try:
    from mymod import (
        myfunc as myfunc_py,
        myfunc_loops2d as myfunc_loops2d_py,
        myfunc_loops3d as myfunc_loops3d_py)
except ImportError:
    pass


try:
    from mymod_default import (
        myfunc as myfunc_default,
        myfunc_loops2d as myfunc_loops2d_default,
        myfunc_loops3d as myfunc_loops3d_default)
except ImportError:
    pass

try:
    from mymod_native import (
        myfunc as myfunc_native,
        myfunc_loops2d as myfunc_loops2d_native,
        myfunc_loops3d as myfunc_loops3d_native)
except ImportError:
    pass

# try:
#     from mymod_omp import (
#         myfunc as myfunc_omp,
#         myfunc_loops2d as myfunc_loops2d_omp,
#         myfunc_loops3d as myfunc_loops3d_omp)
# except ImportError:
#     pass

try:
    from mymod_simd import (
        myfunc as myfunc_simd,
        myfunc_loops2d as myfunc_loops2d_simd,
        myfunc_loops3d as myfunc_loops3d_simd)
except ImportError:
    pass

n2d = 1024
n3d = 128

def rand2d():
    return np.random.rand(n2d, n2d)

def rand3d():
    return np.random.rand(n3d, n3d, n3d)

f2d = rand2d()
f3d = rand3d()

f2d = rand2d() + 1j * rand2d()
f3d = rand3d() + 1j * rand3d()
