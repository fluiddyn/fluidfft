import os
import sys
from concurrent.futures import ThreadPoolExecutor as Pool
from pathlib import Path
from runpy import run_path
from warnings import warn
import sysconfig
import shutil

from setuptools.command.build_ext import build_ext

from setup_configure import (
    configuration,
    lib_dirs_dict,
    lib_flags_dict,
    num_procs,
)
from setup_make import make_extensions, make_pythran_extensions

here = Path(__file__).parent.absolute()

try:
    # pythran > 0.8.6
    from pythran.dist import PythranBuildExt, PythranExtension

    can_import_pythran = True
except ImportError:
    can_import_pythran = False

    PythranBuildExt = object
    PythranExtension = object


class Extension(object):
    def __init__(self, name, sources=None, libraries=None, language=None):
        self.name = name
        self.sources = sources
        self.libraries = libraries
        self.language = language

    def __repr__(self):
        return "setup_build.Extension<{}>".format(self.name)


src_cpp_dir = "src_cpp"
src_cy_dir = "src_cy"
src_cy_dir2d = "fluidfft/fft2d"
src_cy_dir3d = "fluidfft/fft3d"
src_base = "src_cpp/base"
src_cpp_3d = "src_cpp/3d"
src_cpp_2d = "src_cpp/2d"


def create_ext(base_name):

    if base_name.startswith("fft2d"):
        dim = "2d"
        src_cy_dir_dim = src_cy_dir2d
    elif base_name.startswith("fft3d"):
        dim = "3d"
        src_cy_dir_dim = src_cy_dir3d
    else:
        raise ValueError()

    src_cpp_dim = os.path.join(src_cpp_dir, dim)

    source_ends = [".pyx"]
    if base_name.endswith("cufft"):
        source_ends.append(".cu")
    else:
        source_ends.append(".cpp")

    source_files = [base_name + end for end in source_ends]

    base_name = base_name[len("fft2d") :]
    if base_name.startswith("_"):
        base_name = base_name[1:]

    sources = []
    for name_file in source_files:
        if name_file.endswith(".pyx"):
            path = os.path.join(src_cy_dir_dim, name_file)
        else:
            path = os.path.join(src_cpp_dim, name_file)
        sources.append(path)

    sources.extend(
        [
            os.path.join(src_base, "base_fft.cpp"),
            os.path.join(src_cpp_dim, "base_fft" + dim + ".cpp"),
        ]
    )
    if base_name.startswith("mpi"):
        sources.extend(
            [
                os.path.join(src_base, "base_fftmpi.cpp"),
                os.path.join(src_cpp_dim, "base_fft" + dim + "mpi.cpp"),
            ]
        )

    libraries = ["fftw3"]

    if "fftwmpi" in base_name:
        libraries.append("fftw3_mpi")
    elif "pfft" in base_name:
        libraries.extend(["fftw3_mpi", "pfft"])
    elif "p3dfft" in base_name:
        libraries.append("p3dfft")
    elif "cufft" in base_name:
        libraries.extend(["cufft", "mpi_cxx"])

    return Extension(
        name="fluidfft.fft" + dim + "." + base_name,
        sources=sources,
        libraries=libraries,
    )


def base_names_from_config(config):

    from numpy.__config__ import get_info

    try:
        blas_libs = get_info("blas_opt")["libraries"]
        use_mkl_intel = "mkl_intel_lp64" in blas_libs or "mkl_rt" in blas_libs
        # Note: No symbol clash occurs if 'mkl_rt' appears in numpy libraries
        #       instead.
        # P.S.: If 'mkl_rt' is detected, use FFTW libraries, not Intel's MKL/FFTW
        #       implementation.
    except KeyError:
        use_mkl_intel = False

    base_names = []
    if config["fftw3"]["use"]:
        base_names.extend(
            [
                "fft2d_with_fftw1d",
                "fft2d_with_fftw2d",
                "fft2dmpi_with_fftw1d",
                "fft3d_with_fftw3d",
                "fft3dmpi_with_fftw1d",
            ]
        )

    if config["fftw3_mpi"]["use"]:
        if use_mkl_intel:
            warn(
                "When numpy uses mkl (as for example with conda), "
                "there are symbol conflicts between mkl and fftw. "
                "This can lead to a segmentation fault "
                "so we do not build the extensions using fftwmpi."
            )
        else:
            base_names.extend(
                ["fft2dmpi_with_fftwmpi2d", "fft3dmpi_with_fftwmpi3d"]
            )

    if config["cufft"]["use"]:
        base_names.extend(["fft2d_with_cufft"])
        base_names.extend(["fft3d_with_cufft"])

    if config["pfft"]["use"] and not use_mkl_intel:
        base_names.extend(["fft3dmpi_with_pfft"])

    if config["p3dfft"]["use"]:
        base_names.extend(["fft3dmpi_with_p3dfft"])

    return base_names


class FluidFFTBuildExt(build_ext, PythranBuildExt):
    def fluid_make_ext_modules(self):

        # make a python module from cython files
        run_path("src_cy/create_fake_mod_for_doc.py")

        from transonic.dist import make_backend_files

        here = Path(__file__).parent.absolute()
        paths = ["fluidfft/fft2d/operators.py", "fluidfft/fft3d/operators.py"]
        make_backend_files([here / path for path in paths])

        base_names = base_names_from_config(configuration)

        # handle environ (variables) in configuration
        if "environ" in configuration:
            os.environ.update(configuration["environ"])

        from src_cy.make_files_with_mako import make_pyx_files

        make_pyx_files()

        import numpy as np

        include_dirs = [
            src_cy_dir,
            src_cy_dir2d,
            src_cy_dir3d,
            src_base,
            src_cpp_3d,
            src_cpp_2d,
            "include",
            np.get_include(),
        ]

        try:
            import mpi4py
        except ImportError:
            warn(
                "ImportError for mpi4py: "
                "all extensions based on mpi won't be built."
            )
            base_names = [name for name in base_names if "mpi" not in name]
        else:
            if mpi4py.__version__[0] < "2":
                raise ValueError("Please upgrade to mpi4py >= 2.0")
            include_dirs.append(mpi4py.get_include())

        def update_with_config(key):
            cfg = configuration[key]
            if len(cfg["dir"]) > 0:
                path = os.path.join(cfg["dir"], "include")
                if path not in include_dirs:
                    include_dirs.append(path)
            if len(cfg["include_dir"]) > 0:
                path = cfg["include_dir"]
                if path not in include_dirs:
                    include_dirs.append(path)

        if configuration["fftw3"]["use"]:
            update_with_config("fftw3")

        keys = ["pfft", "p3dfft", "cufft"]

        ext_modules = []

        for base_name in base_names:
            ext_modules.append(create_ext(base_name))
            if "fftwmpi" in base_name:
                update_with_config("fftw3_mpi")
            for key in keys:
                if key in base_name:
                    update_with_config(key)

        ext_modules = make_extensions(
            ext_modules,
            include_dirs=include_dirs,
            lib_flags_dict=lib_flags_dict,
            lib_dirs_dict=lib_dirs_dict,
        )

        ext_modules.extend(make_pythran_extensions())

        return ext_modules

    def finalize_options(self):

        # todo: it is not the right place to compile.
        # However, we need to modify self.distribution.ext_modules here.
        # We would have to split this process in
        # 1. create Extension
        # 2. preprocess them (compile and change the source files)

        # 1. should be here and 2. should be in the run method of this class.

        # todo: even better: as in fluidsim

        if not hasattr(self, "fluid_ext_modules"):
            self.distribution.ext_modules = (
                self.fluid_ext_modules
            ) = self.fluid_make_ext_modules()

        super().finalize_options()

    def build_extension(self, ext):
        if len(ext.sources) == 1:
            # we now need to copy Cython ext where setuptools want them
            source = ext.sources[0]
            EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
            if source.endswith(EXT_SUFFIX) and source.startswith("build"):
                relative_path = os.path.sep.join(Path(source).parts[2:])
                abs_path_copy = Path(self.build_lib) / relative_path
                abs_path_copy.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, abs_path_copy)

        if isinstance(ext, PythranExtension):
            return PythranBuildExt.build_extension(self, ext)
        else:
            return build_ext.build_extension(self, ext)

    def build_extensions(self):
        # monkey-patch as in distutils.command.build_ext (parallel)
        self.check_extensions_list(self.extensions)

        to_be_removed = ["-Wstrict-prototypes"]
        starts_forbiden = ["-axMIC_", "-diag-disable:"]

        if hasattr(self.compiler, "compiler_so"):
            self.compiler.compiler_so = [
                key
                for key in self.compiler.compiler_so
                if key not in to_be_removed
                and all([not key.startswith(s) for s in starts_forbiden])
            ]

        # Quick fix to avoid an unexplained bug...
        if sys.version_info[:2] < (3, 7):
            num_procs_ = 1
        else:
            num_procs_ = num_procs
        with Pool(num_procs_) as pool:
            pool.map(self.build_extension, self.extensions)
