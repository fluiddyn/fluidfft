import os
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, PackageLoader


def load_template(filename):
    """Load template file using Jinja2.

    Parameters
    ----------

    filename : str
        Just the filename, without its path.

    Returns
    -------

    jinja2.Template object

    """

    env = Environment(
        loader=PackageLoader("fluidfft_builder", "templates"),
        # undefined=jinja2.StrictUndefined,
        keep_trailing_newline=True,
    )

    return env.get_template(filename)


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

    template_name = f"template{dimension}d.{extension}"

    template = load_template(template_name)

    if not path_output.exists():
        hastomake = True
    else:
        hastomake = modification_date(path_output) < modification_date(
            template.filename
        )

    if hastomake:
        path_output.write_text(
            template.render(
                module_name=module_name,
                class_name=class_name,
                numpy_api=numpy_api,
            )
        )


def main():
    import argparse

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
