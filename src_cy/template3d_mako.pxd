
cdef extern from "fftw3.h":
    ctypedef struct fftw_complex:
        pass

cdef extern from "${module_name}.h":
    cdef cppclass ${class_name}:
        int test()
        const char* bench(int)

        int get_local_size_X()
        int get_local_size_K()

        void get_local_shape_X(int*, int*, int*)
        void get_local_shape_K(int*, int*, int*)

        void get_global_shape_X(int*, int*, int*)
        void get_global_shape_K(int*, int*, int*)

        void get_dimX_K(int*, int*, int*)
        void get_seq_index_first_K(int*, int*)
        
        ${class_name}(int, int, int) except +

        void destroy()

        const char* get_classname()

        void fft(double* fieldX, fftw_complex* fieldK)
        void ifft(fftw_complex* fieldK, double* fieldX)

        double sum_wavenumbers_double(double* fieldK)
        void sum_wavenumbers_complex(fftw_complex* fieldK, fftw_complex* result)
        
        double compute_energy_from_X(double* fieldX)
        double compute_energy_from_K(fftw_complex* fieldK)
        double compute_mean_from_X(double* fieldX)
        double compute_mean_from_K(fftw_complex* fieldK)
