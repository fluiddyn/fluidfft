import argparse

from fluidfft import get_methods, __version__

from pprint import pprint


def main():

    parser = argparse.ArgumentParser(
        prog="fluidfft-get-methods",
        description="Print available FFT methods.",
    )

    parser.add_argument("-V", "--version", action="version", version=__version__)

    parser.add_argument("-d", "--dim", default=None)

    parser.add_argument("-s", "--sequential", action="store_true")
    parser.add_argument("-p", "--parallel", action="store_true")

    args = parser.parse_args()

    dim = args.dim
    if dim is not None:
        dim = int(dim)

    if args.sequential and args.parallel:
        raise ValueError("--sequential and --parallel are incompatible options")
    elif args.sequential:
        sequential = True
    elif args.parallel:
        sequential = False
    else:
        sequential = None

    methods = get_methods(dim, sequential)

    pprint(sorted(methods))
