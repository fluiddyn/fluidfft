import os
from os.path import join
from datetime import datetime

here = os.path.abspath(os.path.dirname(__file__))

path_fluidfft = os.path.abspath(os.path.join(here, "..", "fluidfft"))
path2d = os.path.join(path_fluidfft, "fft2d")
path3d = os.path.join(path_fluidfft, "fft3d")


def load_template(filename):
    """Load template file using Mako or Jinja2.

    Parameters
    ----------

    filename : str
        Just the filename, without its path.

    Returns
    -------

    mako.Template or jinja2.Template object

    """

    try:
        from mako.template import Template

    except ImportError:
        # Use Jinja2 to render Mako style templates
        # See: http://jinja.pocoo.org/docs/2.10/switching/#mako
        from jinja2 import Environment, FileSystemLoader

        env = Environment(
            "<%",
            "%>",
            "${",
            "}",
            "<%doc>",
            "</%doc>",
            "%",
            "##",
            loader=FileSystemLoader(here),
        )
        print("Falling back to Jinja2 as the template library...")
        return env.get_template(filename)
    else:
        return Template(filename=join(here, filename))


def modification_date(filename):
    t = os.path.getmtime(filename)
    return datetime.fromtimestamp(t)


def get_path_files(module_name):
    if module_name.startswith("fft2d"):
        path_package = path2d
    elif module_name.startswith("fft3d"):
        path_package = path3d

    path_pyx = join(path_package, module_name + ".pyx")
    path_pxd = join(path_package, module_name + ".pxd")

    return path_pyx, path_pxd


def make_file(module_name, class_name, numpy_api, templates):

    if module_name.startswith("fft2d"):
        t_pyx = templates["fft2d_pyx"]
        t_pxd = templates["fft2d_pxd"]
    elif module_name.startswith("fft3d"):
        t_pyx = templates["fft3d_pyx"]
        t_pxd = templates["fft3d_pxd"]

    path_pyx, path_pxd = get_path_files(module_name)

    # Generate pyx and pxd files
    for path, template in zip(get_path_files(module_name), (t_pyx, t_pxd)):
        if not os.path.exists(path):
            hastomake = True
        else:
            hastomake = modification_date(path) < modification_date(
                template.filename
            )

        if hastomake:
            with open(path, "w") as f:
                f.write(
                    template.render(
                        module_name=module_name,
                        class_name=class_name,
                        numpy_api=numpy_api,
                    )
                )


variables = (
    ("fft2d_with_fftw1d", "FFT2DWithFFTW1D", "numpy"),
    ("fft2d_with_fftw2d", "FFT2DWithFFTW2D", "numpy"),
    ("fft2d_with_cufft", "FFT2DWithCUFFT", "cupy"),
    ("fft2dmpi_with_fftw1d", "FFT2DMPIWithFFTW1D", "numpy"),
    ("fft2dmpi_with_fftwmpi2d", "FFT2DMPIWithFFTWMPI2D", "numpy"),
    ("fft3d_with_fftw3d", "FFT3DWithFFTW3D", "numpy"),
    ("fft3dmpi_with_fftw1d", "FFT3DMPIWithFFTW1D", "numpy"),
    ("fft3dmpi_with_fftwmpi3d", "FFT3DMPIWithFFTWMPI3D", "numpy"),
    ("fft3dmpi_with_pfft", "FFT3DMPIWithPFFT", "numpy"),
    ("fft3dmpi_with_p3dfft", "FFT3DMPIWithP3DFFT", "numpy"),
    ("fft3d_with_cufft", "FFT3DWithCUFFT", "cupy"),
)


def make_pyx_files():

    templates = {}

    templates["fft2d_pyx"] = load_template("template2d_mako.pyx")
    templates["fft2d_pxd"] = load_template("template2d_mako.pxd")

    templates["fft3d_pyx"] = load_template("template3d_mako.pyx")
    templates["fft3d_pxd"] = load_template("template3d_mako.pxd")

    for module_name, class_name, numpy_api in variables:
        make_file(module_name, class_name, numpy_api, templates)


def _remove(path):
    if os.path.exists(path):
        os.remove(path)


def clean_files():
    for module_name, class_name, _ in variables:
        path_pyx, path_pxd = get_path_files(module_name)
        _remove(path_pyx)
        _remove(path_pxd)
        path_cpp = os.path.splitext(path_pyx)[0] + ".cpp"
        _remove(path_cpp)


if __name__ == "__main__":
    make_pyx_files()
