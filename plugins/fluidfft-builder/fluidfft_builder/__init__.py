from pathlib import Path

from .create_fake_mod_for_doc import create_fake_modules


def print_include_dir():
    src_cpp = Path(__file__).absolute().parent / "src_cpp"
    print(src_cpp)


def print_include_dir_cython():
    include_cy = Path(__file__).absolute().parent / "include_cy"
    print(include_cy)


__all__ = ["create_fake_modules", "print_include_dir", "print_include_dir_cython"]
