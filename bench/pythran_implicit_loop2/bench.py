
import numpy as np

try:
    from mymod_thran import (
        myfunc_ret,
        myfunc,
        myfunc_loops_reshape,
        myfunc_loops2d,
        myfunc_loops3d
    )
except ImportError:
    pass

from mymod import (
    myfunc_ret as myfunc_ret_py,
    myfunc as myfunc_py
)


n2d = 1024
n3d = 128

def rand2d(n0, n1=None):
    if n1 is None:
        n1 = n0
    return np.random.rand(n0, n1)

def rand3d(n0, n1=None, n2=None):
    if n1 is None:
        n1 = n0
    if n2 is None:
        n2 = n0
    return np.random.rand(n0, n1, n2)

f2d = rand2d(n2d, 2*n2d)
f3d = rand3d(n3d, n3d, 2*n3d)

f2d_c = rand2d(n2d) + 1j * rand2d(n2d)
f3d_c = rand3d(n3d) + 1j * rand3d(n3d)
