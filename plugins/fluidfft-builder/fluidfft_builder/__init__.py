from pathlib import Path


def print_include_dir():
    src_cpp = Path(__file__).absolute().parent / "src_cpp"
    print(src_cpp)


def print_include_dir_cython():
    include_cy = Path(__file__).absolute().parent / "include_cy"
    print(include_cy)
