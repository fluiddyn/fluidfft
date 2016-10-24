
from fluidfft import import_fft_class


methods_seq = ['fftw3d']
methods_seq = ['fft3d.with_' + method for method in methods_seq]
classes_seq = [import_fft_class(method) for method in methods_seq]

methods_mpi = ['fftwmpi3d', 'p3dfft', 'pfft']
methods_mpi = ['fft3d.mpi_with_' + method for method in methods_mpi]
classes_mpi = [import_fft_class(method) for method in methods_mpi]
