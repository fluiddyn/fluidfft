import argparse
import os
from datetime import datetime
from importlib import resources
from pathlib import Path
from string import Template


def load_template(filename):
    """Load template file.

    Parameters
    ----------

    filename : str
        Just the filename, without its path.

    Returns
    -------

    A `string.Template` object

    """

    resource = resources.files("fluidfft_builder.templates")

    with resources.as_file((resource / filename)) as file:
        txt = file.read_text()

    return Template(txt)


def modification_date(filename):
    t = os.path.getmtime(filename)
    return datetime.fromtimestamp(t)


def make_file(path_output, class_name, numpy_api="numpy"):
    if not class_name.startswith("FFT"):
        raise ValueError('not module_name.startswith("fft")')

    dimension = class_name[3]

    if dimension not in "23":
        raise ValueError('dimension not in "23"')

    path_output = Path(path_output)
    name_output = path_output.name
    module_name, extension = name_output.split(".")

    if extension == "pxd":
        module_name = module_name[5:]
        if module_name.startswith("_"):
            module_name = module_name[1:]

    cpp_name = f"fft{dimension}d"
    if module_name.startswith("mpi"):
        cdef_public_mpi_comm = "cdef public MPI.Comm comm"
        include_base_mpi_pyx = "include 'base_mpi.pyx'"
    else:
        cdef_public_mpi_comm = ""
        include_base_mpi_pyx = ""
        cpp_name += "_"

    cpp_name += module_name

    template_name = f"template{dimension}d.{extension}"
    template = load_template(template_name)

    content = template.substitute(
        {
            "module_name": module_name,
            "cpp_name": cpp_name,
            "class_name": class_name,
            "numpy_api": numpy_api,
            "cdef_public_mpi_comm": cdef_public_mpi_comm,
            "include_base_mpi_pyx": include_base_mpi_pyx,
        }
    )

    if not path_output.exists():
        hastomake = True
    else:
        content_old = path_output.read_text(encoding="utf8")
        hastomake = content != content_old

    if hastomake:
        path_output.write_text(content, encoding="utf8")


def main():

    parser = argparse.ArgumentParser(
        prog="fluidfft-builder-make-file",
        description="Make Cython files for fluidfft templates",
    )

    parser.add_argument("name_output", type=str)

    parser.add_argument("class_name", type=str)

    args = parser.parse_args()
    print(args)
    # raise ValueError(f"{args} {CURRENT_DIR=}")

    make_file(args.name_output, args.class_name)
