
import numpy as np

from vectprod_native import vectprod as vectprod_native
from vectprod_default import vectprod as vectprod_default
from vectprod_omp import vectprod as vectprod_omp
from vectprod import vectprod as vectprod_py

from vectprod_native import vectprod_explicitloop as vectprod1_native
from vectprod_default import vectprod_explicitloop as vectprod1_default
from vectprod_omp import vectprod_explicitloop as vectprod1_omp
from vectprod import vectprod_explicitloop as vectprod1_py

from vectprod_native import vectprod_inplace as vectprod2_native
from vectprod_default import vectprod_inplace as vectprod2_default
from vectprod_omp import vectprod_inplace as vectprod2_omp
from vectprod import vectprod_inplace as vectprod2_py


from fluidfft.fft3d.util_pythran import vector_product as vectprod_fft

n0 = n1 = n2 = 128

shape = (n0, n1, n2)

a0 = np.ones(shape)
a1 = np.ones(shape)
a2 = np.ones(shape)
a3 = np.ones(shape)
a4 = np.ones(shape)
a5 = np.ones(shape)
