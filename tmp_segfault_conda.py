
"""With OpenMPI:
python -c "import mpi4py as p; print(p.get_config())"
{'mpicc': '/usr/bin/mpicc', 'mpicxx': '/usr/bin/mpicxx', 'mpif77': '/usr/bin/mpif77', 'mpif90': '/usr/bin/mpif90'}


- If numpy is imported before, we get Segmentation fault.
- otherwise, it works fine...

For the unittest we also have a segmentation fault related to numpy:

/home/users/augier3pi/opt/miniconda3/lib/python3.6/site-packages/numpy/core/../../../../libmkl_intel_lp64.so(fftw_execute+0xf) [0x7f0293525e7f]
/home/users/augier3pi/Dev/fluidfft/fluidfft/fft2d/mpi_with_fftwmpi2d.cpython-36m-x86_64-linux-gnu.so(_ZN21FFT2DMPIWithFFTWMPI2D3fftEPdPCd+0x81) [0x7f027a101f31]

See:
https://stackoverflow.com/questions/22004131/is-there-symbol-conflict-when-loading-two-shared-libraries-with-a-same-symbol
https://stackoverflow.com/questions/678254/what-should-i-do-if-two-libraries-provide-a-function-with-the-same-name-generati

Is there a way to specify that we fluidfft.fft2d.mpi_with_fftwmpi2d to use
/usr/lib/x86_64-linux-gnu/libfftw3_mpi.so.3 and not libmkl_intel_lp64.so ?

ldd fluidfft/fft2d/mpi_with_fftwmpi2d.cpython-36m-x86_64-linux-gnu.so
	linux-vdso.so.1 (0x00007ffdb6ddc000)
	libfftw3.so.3 => /usr/lib/x86_64-linux-gnu/libfftw3.so.3 (0x00007fbd1349d000)
	libfftw3_mpi.so.3 => /usr/lib/x86_64-linux-gnu/libfftw3_mpi.so.3 (0x00007fbd13286000)

nm ~/opt/miniconda3/lib/python3.6/site-packages/numpy/core/../../../../libmkl_intel_lp64.so | grep fftw_execute

"""

# Uncomment this import to get the segmentation fault.
# With this import, mpi_with_fftw1d is also faster because it uses mlk fft.
# import numpy as np

from fluidfft.fft2d.mpi_with_fftw1d import FFT2DMPIWithFFTW1D as cls
# from fluidfft.fft2d.mpi_with_fftwmpi2d import FFT2DMPIWithFFTWMPI2D as cls

import numpy as np


o = cls(1024, 1024)

# o.run_tests()
o.run_benchs(100)

# a = np.random.random(o.get_local_size_X()).reshape(
#     o.get_shapeX_loc())
# afft = o.fft(a)
# a = o.ifft(afft)
# afft = o.fft(a)

# EX = o.compute_energy_from_X(a)
# EK = o.compute_energy_from_K(afft)

# assert np.allclose(EX, EK)
print('Works well')
