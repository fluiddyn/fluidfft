
import numpy as np


# pythran export vectprod(
#     float64[][][], float64[][][], float64[][][],
#     float64[][][], float64[][][], float64[][][])

def vectprod(ax, ay, az, bx, by, bz):
    return (ay * bz - az * by,
            az * bx - ax * bz,
            ax * by - ay * bx)


# pythran export vectprod_inplace_noloop(
#     float64[][][], float64[][][], float64[][][],
#     float64[][][], float64[][][], float64[][][])

def vectprod_inplace_noloop(ax, ay, az, bx, by, bz):
    # without loop, we need copies!
    tmpx = bx.copy()
    tmpy = by.copy()
    tmpz = bz.copy()

    bx[:] = ay * tmpz - az * tmpy
    by[:] = az * tmpx - ax * tmpz
    bz[:] = ax * tmpy - ay * tmpx

    return bx, by, bz


# pythran export vectprod_explicitloop(
#     float64[][][], float64[][][], float64[][][],
#     float64[][][], float64[][][], float64[][][])

def vectprod_explicitloop(ax, ay, az, bx, by, bz):
    resultx = np.empty_like(ax)
    resulty = np.empty_like(ax)
    resultz = np.empty_like(ax)

    n0, n1, n2 = ax.shape

    for i0 in range(n0):
        for i1 in range(n1):
            for i2 in range(n2):
                elem_ax = ax[i0, i1, i2]
                elem_ay = ay[i0, i1, i2]
                elem_az = az[i0, i1, i2]
                elem_bx = bx[i0, i1, i2]
                elem_by = by[i0, i1, i2]
                elem_bz = bz[i0, i1, i2]

                resultx[i0, i1, i2] = elem_ay * elem_bz - elem_az * elem_by
                resulty[i0, i1, i2] = elem_az * elem_bx - elem_ax * elem_bz
                resultz[i0, i1, i2] = elem_ax * elem_by - elem_ay * elem_bx

    return resultx, resulty, resultz


# pythran export vectprod_inplace(
#     float64[][][], float64[][][], float64[][][],
#     float64[][][], float64[][][], float64[][][])

def vectprod_inplace(ax, ay, az, bx, by, bz):

    n0, n1, n2 = ax.shape

    for i0 in range(n0):
        for i1 in range(n1):
            for i2 in range(n2):
                elem_ax = ax[i0, i1, i2]
                elem_ay = ay[i0, i1, i2]
                elem_az = az[i0, i1, i2]
                elem_bx = bx[i0, i1, i2]
                elem_by = by[i0, i1, i2]
                elem_bz = bz[i0, i1, i2]

                bx[i0, i1, i2] = elem_ay * elem_bz - elem_az * elem_by
                by[i0, i1, i2] = elem_az * elem_bx - elem_ax * elem_bz
                bz[i0, i1, i2] = elem_ax * elem_by - elem_ay * elem_bx

    return bx, by, bz
