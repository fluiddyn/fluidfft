
# pythran export _vgradv_from_v2(
#     float64[][][], float64[][][], float64[][][],
#     complex128[][][], complex128[][][], complex128[][][],
#     float64[][][], float64[][][], float64[][][],
#     function_to_be_called_from_python_interpreter -> float64[][][])


def _vgradv_from_v2(vx, vy, vz, vx_fft, vy_fft, vz_fft, Kx, Ky, Kz, ifft3d):
    px_vx_fft = 1j * Kx * vx_fft
    py_vx_fft = 1j * Ky * vx_fft
    pz_vx_fft = 1j * Kz * vx_fft

    px_vy_fft = 1j * Kx * vy_fft
    py_vy_fft = 1j * Ky * vy_fft
    pz_vy_fft = 1j * Kz * vy_fft

    px_vz_fft = 1j * Kx * vz_fft
    py_vz_fft = 1j * Ky * vz_fft
    pz_vz_fft = 1j * Kz * vz_fft

    vgradvx = (
        vx * ifft3d(px_vx_fft) + vy * ifft3d(py_vx_fft) + vz * ifft3d(pz_vx_fft)
    )

    vgradvy = (
        vx * ifft3d(px_vy_fft) + vy * ifft3d(py_vy_fft) + vz * ifft3d(pz_vy_fft)
    )

    vgradvz = (
        vx * ifft3d(px_vz_fft) + vy * ifft3d(py_vz_fft) + vz * ifft3d(pz_vz_fft)
    )

    return vgradvx, vgradvy, vgradvz


"""If we can not use something like the syntax

'function_to_be_called_from_python_interpreter -> float64[][][]'

we have to slit the function _vgradv_from_v2 in parts that can be pythranized
and other that can't be pythranized (here we are lucky, it's quite simple!).

If we have a lot of functions like _vgradv_from_v2, it's a lot of work and the
code becomes much less simple. We do not want that!

"""


# pythran export part0(
#     complex128[][][], complex128[][][], complex128[][][],
#     float64[][][], float64[][][], float64[][][])


def part0(vx_fft, vy_fft, vz_fft, Kx, Ky, Kz):

    px_vx_fft = 1j * Kx * vx_fft
    py_vx_fft = 1j * Ky * vx_fft
    pz_vx_fft = 1j * Kz * vx_fft

    px_vy_fft = 1j * Kx * vy_fft
    py_vy_fft = 1j * Ky * vy_fft
    pz_vy_fft = 1j * Kz * vy_fft

    px_vz_fft = 1j * Kx * vz_fft
    py_vz_fft = 1j * Ky * vz_fft
    pz_vz_fft = 1j * Kz * vz_fft

    return (
        px_vx_fft,
        py_vx_fft,
        pz_vx_fft,
        px_vy_fft,
        py_vy_fft,
        pz_vy_fft,
        px_vz_fft,
        py_vz_fft,
        pz_vz_fft,
    )


# pythran export part1(
#     float64[][][], float64[][][], float64[][][],
#     float64[][][], float64[][][], float64[][][],
#     float64[][][], float64[][][], float64[][][],
#     float64[][][], float64[][][], float64[][][])


def part1(
    vx, vy, vz, px_vx, py_vx, pz_vx, px_vy, py_vy, pz_vy, px_vz, py_vz, pz_vz
):

    vgradvx = (vx * px_vx + vy * py_vx + vz * pz_vx)

    vgradvy = (vx * px_vy + vy * py_vy + vz * pz_vy)

    vgradvz = (vx * px_vz + vy * py_vz + vz * pz_vz)

    return vgradvx, vgradvy, vgradvz


"""The pure python function that we would have to write with the pythranized
function part0 and part1:

def _vgradv_from_v3(vx, vy, vz, vx_fft, vy_fft, vz_fft,
                    Kx, Ky, Kz, ifft3d):

    (px_vx_fft, py_vx_fft, pz_vx_fft,
     px_vy_fft, py_vy_fft, pz_vy_fft,
     px_vz_fft, py_vz_fft, pz_vz_fft) = part0(vx_fft, vy_fft, vz_fft,
                                              Kx, Ky, Kz)

    px_vx = ifft3d(px_vx_fft)
    py_vx = ifft3d(py_vx_fft)
    pz_vx = ifft3d(pz_vx_fft)

    px_vy = ifft3d(px_vy_fft)
    py_vy = ifft3d(py_vy_fft)
    pz_vy = ifft3d(pz_vy_fft)

    px_vz = ifft3d(px_vz_fft)
    py_vz = ifft3d(py_vz_fft)
    pz_vz = ifft3d(pz_vz_fft)

    return part1(vx, vy, vz,
                 px_vx, py_vx, pz_vx,
                 px_vy, py_vy, pz_vy,
                 px_vz, py_vz, pz_vz)

"""
