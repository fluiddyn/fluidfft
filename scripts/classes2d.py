
from fluidfft import import_fft_class

methods_seq = ['fftw1d', 'fftw2d']
methods_seq = ['fft2d.with_' + method for method in methods_seq]
classes_seq = [import_fft_class(method) for method in methods_seq]

methods_mpi = ['fftwmpi2d', 'fftw1d']
methods_mpi = ['fft2d.mpi_with_' + method for method in methods_mpi]
classes_mpi = [import_fft_class(method) for method in methods_mpi]
