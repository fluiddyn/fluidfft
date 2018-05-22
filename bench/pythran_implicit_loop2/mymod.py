

# pythran export myfunc_ret(float64[][], float64[][])
# pythran export myfunc_ret(complex128[][], complex128[][])
# pythran export myfunc_ret(float64[][][], float64[][][])
# pythran export myfunc_ret(complex128[][][], complex128[][][])


def myfunc_ret(a, b):
    return (a**2 + b**3 + 2) / 5.


# pythran export myfunc(float64[][], float64[][])
# pythran export myfunc(complex128[][], complex128[][])
# pythran export myfunc(float64[][][], float64[][][])
# pythran export myfunc(complex128[][][], complex128[][][])

def myfunc(a, b):
    a[:] = myfunc_ret(a, b)

# pythran export myfunc_loops_reshape(float64[][], float64[][])
# pythran export myfunc_loops_reshape(complex128[][], complex128[][])
# pythran export myfunc_loops_reshape(float64[][][], float64[][][])
# pythran export myfunc_loops_reshape(complex128[][][], complex128[][][])


def myfunc_loops_reshape(a, b):
    if a.shape == b.shape:
        n = a.size
        a = a.reshape((-1,))
        b = b.reshape((-1,))
        for i in range(n):
            a[i] = myfunc_ret(a[i], b[i])
    else:
        myfunc(a, b)


# pythran export myfunc_loops2d(float64[][], float64[][])
# pythran export myfunc_loops2d(complex128[][], complex128[][])

def myfunc_loops2d(a, b):
    n0, n1 = a.shape
    for i0 in range(n0):
        for i1 in range(n1):
            a[i0, i1] = myfunc_ret(a[i0, i1], b[i0, i1])


# pythran export myfunc_loops3d(float64[][][], float64[][][])
# pythran export myfunc_loops3d(complex128[][][], complex128[][][])

def myfunc_loops3d(a, b):
    n0, n1, n2 = a.shape
    for i0 in range(n0):
        for i1 in range(n1):
            for i2 in range(n2):
                a[i0, i1, i2] = myfunc_ret(a[i0, i1, i2], b[i0, i1, i2])
