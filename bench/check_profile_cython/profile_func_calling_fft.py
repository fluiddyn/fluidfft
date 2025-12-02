import pstats
import cProfile

from fluidfft.fft3d import get_classes_seq


N = 128

classes = get_classes_seq()
print(classes)


def init_oper(fft_class):
    return fft_class(N, N, N)


def run(fft_cls):
    print(fft_cls)
    o = init_oper(fft_cls)
    o.run_tests()
    o.run_benchs()
    o.run_benchs()


def main():

    # no issue with profiling
    name = "fft3d.with_pyfftw"

    # issue with profiling (before fluidfft-builder 0.0.3): fft functions do not appear
    name = "fft3d.with_fftw3d"
    cls = classes[name]
    run(cls)


# main()

cProfile.runctx("main()", globals(), locals(), "profile.pstats")

s = pstats.Stats("profile.pstats")
# s.strip_dirs().sort_stats('time').print_stats(16)
s.sort_stats("time").print_stats(12)
