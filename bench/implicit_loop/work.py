
import numpy as np

from mymod_default import myfunc2

shape = (1024,)*2

a = np.ones(shape, dtype=np.float64)

for i in range(2000):
    myfunc2(a)
