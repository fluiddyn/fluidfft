
import numpy as np
from time import time

from mymod_openmp import myfunc2

shape = (1024,)*2

a = np.ones(shape, dtype=np.float64)

t_start = time()
for i in range(5000):
    myfunc2(a)
print(f'done in {time() - t_start:.2f} s')
