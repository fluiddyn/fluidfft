
cdef extern from "base_fft.h":
    ctypedef struct mycomplex:
        pass

cdef extern from "${cpp_name}.h":
    cdef cppclass ${class_name}:
        int test()
        void bench(int, double*)

        int get_local_size_X()
        int get_local_size_K()

        void get_local_shape_X(int*, int*)
        void get_local_shape_K(int*, int*)

        void get_shapeX_seq(int*, int*)
        void get_shapeK_seq(int*, int*)

        ${class_name}(int, int) except +

        void destroy()

        const char* get_classname()

        void fft(double* fieldX, mycomplex* fieldK)
        void ifft(mycomplex* fieldK, double* fieldX)

        double compute_energy_from_X(double* fieldX)
        double compute_energy_from_K(mycomplex* fieldK)
        double sum_wavenumbers(double* fieldK)
        double compute_mean_from_X(double* fieldX)
        double compute_mean_from_K(mycomplex* fieldK)

        char get_is_transposed()
        ptrdiff_t get_local_X0_start()
        ptrdiff_t get_local_K0_start()
