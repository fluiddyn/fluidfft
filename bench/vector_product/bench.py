
import numpy as np

from vectprod_native import vectprod as vectprod_native
from vectprod_default import vectprod as vectprod_default
from vectprod import vectprod as vectprod_py
from vectprod_simd import vectprod as vectprod_simd

from vectprod_native import vectprod_explicitloop as vectprod1_native
from vectprod_default import vectprod_explicitloop as vectprod1_default
from vectprod import vectprod_explicitloop as vectprod1_py
from vectprod_simd import vectprod_explicitloop as vectprod1_simd

from vectprod_native import vectprod_inplace as vectprod2_native
from vectprod_default import vectprod_inplace as vectprod2_default
from vectprod import vectprod_inplace as vectprod2_py
from vectprod_simd import vectprod_inplace as vectprod2_simd

from vectprod_native import vectprod_inplace_noloop as vectprod3_native
from vectprod_default import vectprod_inplace_noloop as vectprod3_default
from vectprod import vectprod_inplace_noloop as vectprod3_py
from vectprod_simd import vectprod_inplace_noloop as vectprod3_simd

try:
    from vectprod_omp import vectprod as vectprod_omp
    from vectprod_omp import vectprod_explicitloop as vectprod1_omp
    from vectprod_omp import vectprod_inplace as vectprod2_omp
    from vectprod_omp import vectprod_inplace_noloop as vectprod3_omp
except ImportError:
    pass

try:
    from fluidfft.fft3d.util_pythran import vector_product as vectprod_fft
except ImportError:
    pass

n0 = n1 = n2 = 128

shape = (n0, n1, n2)

rand = np.random.randn

a0 = rand(*shape)
a1 = rand(*shape)
a2 = rand(*shape)
a3 = rand(*shape)
a4 = rand(*shape)
a5 = rand(*shape)
