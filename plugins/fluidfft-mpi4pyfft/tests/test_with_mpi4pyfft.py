from fluidfft.fft3d.testing import make_testop_functions


from fluidfft import import_fft_class


class Tests3D(unittest.TestCase):
    pass


def complete_class(name, cls):
    tests = make_testop_functions(name, cls)

    for key, test in tests.items():
        setattr(Tests3D, "test_operator3d_{}_{}".format(name, key), test)


methods = ["fft3d.mpi_with_mpi4pyfft", "fft3d.mpi_with_mpi4pyfft_slab"]
for method in methods:
    name = method.split(".")[1]
    cls = import_fft_class(method)
    complete_class(name, cls)
