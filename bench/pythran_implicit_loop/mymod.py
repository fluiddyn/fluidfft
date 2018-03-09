
import numpy as np


# pythran export myfunc(float64[][])
# pythran export myfunc(complex128[][])
# pythran export myfunc(float64[][][])
# pythran export myfunc(complex128[][][])

def myfunc(a):
    return (a**2 + a**3 + 2) / 5.


# pythran export myfunc_loops2d(float64[][])
# pythran export myfunc_loops2d(complex128[][])

def myfunc_loops2d(aa):
    n0, n1 = aa.shape
    result = np.empty_like(aa)
    for i0 in range(n0):
        for i1 in range(n1):
            a = aa[i0, i1]
            result[i0, i1] = myfunc(a)
    return result


# pythran export myfunc_loops3d(float64[][][])
# pythran export myfunc_loops3d(complex128[][][])

def myfunc_loops3d(aa):
    n0, n1, n2 = aa.shape
    result = np.empty_like(aa)
    for i0 in range(n0):
        for i1 in range(n1):
            for i2 in range(n2):
                a = aa[i0, i1, i2]
                result[i0, i1, i2] = myfunc(a)
    return result
