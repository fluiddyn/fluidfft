

import numpy as np

# pythran export myfunc(
#     float[], int,
#     function_to_be_called_by_python_interpreter -> float64[]
#     function_to_be_called_by_python_interpreter -> complex128[])

def myfunc(a, n, func0, func1):

    c = a**2 + a**3 + 1

    tmp = np.empty(n)
    for i in range(n):
        tmp[i] = func0(i*c)

    ret = tmp**4 + 2*func1(a)

    return ret
