"""Maker for fluidfft
=====================

The compilation process of the normal build_ext is not adapted for this project
since many files are common to different extensions but can be computed only
once.

All files are compiled manually and we launch the compilations asynchronously.

Notes
-----

- compilation:

  $CC $OPT $BASECFLAGS $CONFIGURE_CFLAGS $CCSHARED \
    $(sys.prefix)/include/python$('.'.join(sys.version_info[:2])) \
    -c path/source.cpp -o build/temp.linux-x86_64-2.7/path/source.o

- linker:

  $CXX -shared \
    -Wl,-rpath=$(sys.prefix)/lib/python$('.'.join(sys.version_info[:2])) \
    $(list object files) -L/opt/python/2.7.9/lib -lpython2.7 \
    -o build/temp.linux-x86_64-2.7/path/to/mod.so

  $CXX -shared ~$(LDSHARED) $(CONFIGURE_LDFLAGS) $(CONFIGURE_CFLAGS) \
  $(list object files)

"""

import sys
from time import sleep
import os
from datetime import datetime
import subprocess
from copy import copy
import multiprocessing
import warnings
from pathlib import Path

from setuptools import Extension as SetuptoolsExtension

from setup_configure import (
    build_needs_mpi4py,
    get_distconfig,
    num_procs,
    TRANSONIC_BACKEND,
)


distconfig = get_distconfig()
here = Path(__file__).parent.absolute()


if build_needs_mpi4py:
    mpicxx_compile_words = []
    try:
        # does not work with mpich2 (used by default by anaconda)
        mpicxx_compile_words = (
            subprocess.check_output(["mpicxx", "-showme:compile"])
            .decode()
            .split()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            subprocess.call(["CC", "--version"])
            mpicxx_compile_words = (
                subprocess.check_output(["CC", "--cray-print-opts=all"])
                .decode()
                .split()
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            warnings.warn(
                "Unable to find MPI compile flags."
                "Setting mpicxx_compile_words=[]"
            )


def modification_date(filename):
    t = os.path.getmtime(filename)
    return datetime.fromtimestamp(t)


class CompilationError(Exception):
    def __init__(self, command, out, err):
        try:
            out = out.decode()
        except UnicodeDecodeError:
            pass
        try:
            err = err.decode()
        except UnicodeDecodeError:
            pass

        self.message = "\ncommand:\n{}\nstdout:\n {}\nstderr:\n{}".format(
            command, out, err
        )
        super(CompilationError, self).__init__()

    def __str__(self):
        return super(CompilationError, self).__str__() + self.message


def has_to_build(output_file, input_files):
    if not os.path.exists(output_file):
        return True
    mod_date_output = modification_date(output_file)
    for input_file in input_files:
        if mod_date_output < modification_date(input_file):
            return True
    return False


class CommandsRunner(object):
    """Run command in parallel (with processes)"""

    def __init__(self, commands):
        self.commands = commands
        self._processes = []

    def run(self):
        while len(self.commands) != 0:
            if len(self._processes) < num_procs:
                command = self.commands.pop()
                self._launch_process(command)

            sleep(0.1)
            self._check_processes()

        while len(self._processes) != 0:
            sleep(0.1)
            self._check_processes()

    def _launch_process(self, command):
        print("launching command:\n" + " ".join(command))
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        process.command = " ".join(command)
        process.stderr_log = b""
        self._processes.append(process)

    def _check_processes(self):
        for process in copy(self._processes):
            # hack (if num_procs == 1) to avoid bug (Python)
            # https://stackoverflow.com/questions/11937028/poll-method-of-subprocess-popen-returning-none-for-long-process
            if num_procs == 1:
                process.stderr_log += process.stderr.read()
            if process.poll() is not None:
                self._processes.remove(process)
                if process.returncode != 0:
                    out, err = process.communicate()
                    process.stderr_log += err
                    lines_err = process.stderr_log.split(b"\n")
                    if len(lines_err) > 40:
                        log_err = b"\n".join(lines_err[:40])
                        file_name = (
                            "/tmp/fluidfft_build_error_log_" + f"pid{process.pid}"
                        )
                        with open(file_name, "w") as file:
                            file.write(process.stderr_log.decode())
                        log_err += (
                            b"\n[...]\n\nend of error log in "
                            + file_name.encode()
                        )
                    else:
                        log_err = process.stderr_log

                    raise CompilationError(process.command, out, log_err)


class FunctionsRunner(CommandsRunner):
    """Run function calls in parallel (with multiprocessing)"""

    def _launch_process(self, command):
        func, args, kwargs = command
        print(
            "launching command:\n"
            "{}(\n    *args={},\n    **kwargs={})".format(
                func.__name__, args, kwargs
            )
        )
        process = multiprocessing.Process(target=func, args=args, kwargs=kwargs)
        process.start()
        process.command = command
        self._processes.append(process)

    def _check_processes(self):
        for process in copy(self._processes):
            if process.exitcode is not None:
                self._processes.remove(process)


def make_function_cpp_from_pyx(
    cpp_file,
    pyx_file,
    include_dirs=None,
    compiler_directives=None,
    full_module_name=None,
):
    if compiler_directives is None:
        compiler_directives = {}

    path_dir = os.path.split(cpp_file)[0]
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)

    if not has_to_build(cpp_file, (pyx_file,)):
        return

    from Cython.Compiler.Main import (
        compile_single,
        default_options,
        CompilationOptions,
    )

    options = CompilationOptions(
        default_options,
        include_path=include_dirs,
        output_file=cpp_file,
        cplus=True,
        compiler_directives=compiler_directives,
        compile_time_env={"MPI4PY": build_needs_mpi4py},
    )

    # return (func, args, kwargs)
    return (
        compile_single,
        (pyx_file,),
        {"options": options, "full_module_name": full_module_name},
    )


def make_command_obj_from_cpp(
    obj_file, cpp_file, include_dirs=None, options=None
):
    path_dir = os.path.split(obj_file)[0]
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)

    if not has_to_build(obj_file, (cpp_file,)):
        return

    _distconfig = copy(distconfig)

    if options is not None:
        _distconfig.update({k: v for k, v in options.items() if v is not None})

    keys = ["CXX", "OPT", "BASECFLAGS", "CFLAGS", "CCSHARED"]
    command = " ".join([_distconfig[k] for k in keys])

    if cpp_file.endswith(".cu"):
        NVCC = os.getenv("NVCC", "nvcc")
        command = (
            NVCC + " -m64 "
            "-gencode arch=compute_30,code=sm_30 "
            "-gencode arch=compute_32,code=sm_32 "
            "-gencode arch=compute_35,code=sm_35 "
            "-gencode arch=compute_50,code=sm_50 "
            "-gencode arch=compute_50,code=compute_50 -Xcompiler -fPIC"
        )

    command = [
        w
        for w in command.split()
        if w not in ["-g", "-DNDEBUG", "-Wstrict-prototypes"]
    ]

    if build_needs_mpi4py:
        if cpp_file.endswith(".cu"):
            for word in set(mpicxx_compile_words).difference(command):
                if word == "-pthread":
                    continue
                command.append(word)
        else:
            mpicxx = _distconfig["MPICXX"].split()
            mpicxx.extend(command[1:])
            command = mpicxx

    includepy = _distconfig["INCLUDEPY"]

    includedir = os.path.split(includepy)[0]
    if os.path.split(includedir)[-1] == "include":
        includepy = [includepy, includedir]
    else:
        includepy = [includepy]
    if include_dirs is None:
        include_dirs = includepy
    else:
        include_dirs = includepy + include_dirs

    command += ["-I" + incdir for incdir in include_dirs]
    command += ["-c", cpp_file, "-o", obj_file]
    return command


def make_command_ext_from_objs(
    ext_file, obj_files, lib_flags=None, lib_dirs=None
):
    if not has_to_build(ext_file, obj_files):
        return

    path_dir = os.path.split(ext_file)[0]
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)

    cxx = distconfig["CXX"]
    ldshared = distconfig["LDSHARED"]

    if "mpi_with_p3dfft" in ext_file or "mpi_with_pfft" in ext_file:
        # remove an option incompatible with these libraries
        if "-Wl,--as-needed" in ldshared:
            ldshared = ldshared.replace("-Wl,--as-needed", "-Wl,--no-as-needed")
        else:
            ldshared += " -Wl,--no-as-needed"

    command = [w for w in ldshared.split() if w not in ["-g"]]

    if build_needs_mpi4py:
        command[0] = os.getenv("MPICXX", "mpicxx")
    else:
        command[0] = cxx.split()[0]

    if "cufft" in ext_file:
        NVCC = os.getenv("NVCC", "nvcc")
        command = (NVCC + " -Xcompiler -pthread -shared").split()

    command += obj_files + ["-o", ext_file]
    if lib_dirs is not None:
        command.extend(["-L" + lib_dir for lib_dir in lib_dirs])

    if lib_flags is not None:
        command.extend(["-l" + lib for lib in lib_flags])

    return command


def make_extensions(
    extensions,
    include_dirs=None,
    lib_dirs_dict=None,
    lib_flags_dict=None,
    **options,
):
    if all(
        command not in sys.argv
        for command in [
            "build_ext",
            "install",
            "develop",
            "bdist_wheel",
            "bdist_egg",
            "editable_wheel",
        ]
    ):
        return []

    sources = set()
    for ext in extensions:
        print("make_extensions: ", ext)
        sources.update(ext.sources)

    # prepare a dictionary listing all files
    files = {}
    extension_files = ["pyx", "cpp"]

    for ext in extension_files:
        files[ext] = []

    for source in sources:
        for ext in extension_files:
            if source.endswith("." + ext):
                files[ext].append(source)
        if source.endswith(".cu"):
            files["cpp"].append(source)
    files["o"] = []

    # Enable linetrace for Cython extensions while using tox
    if os.getenv("TOXENV") is not None:
        warnings.warn(
            "Enabling linetrace for coverage tests. Extensions can be really "
            "slow. Recompile for practical use."
        )
        compiler_directives = {"linetrace": True}
        cython_options = {"CFLAGS": " -DCYTHON_TRACE_NOGIL=1"}
        if "CFLAGS" in options:
            options["CFLAGS"] += cython_options["CFLAGS"]
        else:
            options.update(cython_options)
    else:
        compiler_directives = {}

    # cythonize .pyx files if needed
    commands = []
    for pyx_file in files["pyx"]:
        if "cufft" in pyx_file:
            cpp_ext = ".cu"
        else:
            cpp_ext = ".cpp"
        cpp_file = os.path.splitext(pyx_file)[0] + cpp_ext

        full_module_name = None
        for ext in extensions:
            if pyx_file in ext.sources:
                full_module_name = ext.name

        command = make_function_cpp_from_pyx(
            cpp_file,
            pyx_file,
            full_module_name=full_module_name,
            include_dirs=include_dirs,
            compiler_directives=compiler_directives,
        )
        if command is not None:
            commands.append(command)

        files["cpp"].append(cpp_file)

    FunctionsRunner(commands).run()

    # compile .cpp files if needed
    commands = []
    for path in files["cpp"]:
        result = os.path.join(
            distconfig["PATH_TMP"], os.path.splitext(path)[0] + ".o"
        )
        command = make_command_obj_from_cpp(result, path, include_dirs, options)
        if command is not None:
            commands.append(command)
        files["o"].append(result)

    CommandsRunner(commands).run()

    extensions_out = []

    # link .o files to produce the .so files if needed
    path_base_output = distconfig["PATH_LIB"]
    files["so"] = []
    commands = []
    for ext in extensions:
        suffix = distconfig["EXT_SUFFIX"]
        result = os.path.join(
            path_base_output, ext.name.replace(".", os.path.sep) + suffix
        )
        objects = [
            os.path.join(
                distconfig["PATH_TMP"], os.path.splitext(source)[0] + ".o"
            )
            for source in ext.sources
        ]

        extensions_out.append(SetuptoolsExtension(ext.name, sources=[result]))

        lib_dirs = []
        lib_flags = []
        for lib in ext.libraries:
            if lib_dirs_dict is not None and lib in lib_dirs_dict:
                lib_dirs.append(lib_dirs_dict[lib])
            if lib_flags_dict is not None and lib in lib_flags_dict:
                lib_flags.extend(lib_flags_dict[lib])
            else:
                lib_flags.append(lib)

        command = make_command_ext_from_objs(
            result, objects, lib_dirs=lib_dirs, lib_flags=lib_flags
        )
        if command is not None:
            commands.append(command)
        files["so"].append(result)

    # wait for linking
    CommandsRunner(commands).run()
    return extensions_out


def make_pythran_extensions():
    modules = []
    for root, dirs, files in os.walk("src/fluidfft"):
        path_dir = Path(root)
        for name in files:
            if path_dir.name == "__pythran__" and name.endswith(".py"):
                path = os.path.join(root, name)
                modules.append(path[4:].replace(os.path.sep, ".").split(".py")[0])

    import numpy as np

    if TRANSONIC_BACKEND != "pythran":
        return []

    from pythran.dist import PythranExtension

    develop = "develop" in sys.argv or "editable_wheel" in sys.argv
    extensions = []
    for mod in modules:
        base_file = mod.replace(".", os.path.sep)
        py_file = "src/" + base_file + ".py"
        # warning: does not work on Windows (?)
        suffix = distconfig["EXT_SUFFIX"]
        bin_file = base_file + suffix
        if (
            not develop
            or not os.path.exists(bin_file)
            or modification_date(bin_file) < modification_date(py_file)
        ):
            pext = PythranExtension(mod, [py_file])
            # bug pythran extension...
            pext.include_dirs.append(np.get_include())
            pext.extra_compile_args.append("-O3")

            # by default, compile with -march=native
            if not os.getenv("FLUIDFFT_NO_CARCH", False):
                compile_arch = os.getenv("CARCH", "native")
                pext.extra_compile_args.append(f"-march={compile_arch}")
            else:
                print("FLUIDFFT_NO_CARCH")

            if not os.getenv("FLUIDDYN_NO_XSIMD", False):
                pext.extra_compile_args.append("-DUSE_XSIMD")
            else:
                print("FLUIDDYN_NO_XSIMD")

            # pext.extra_compile_args.append('-fopenmp')
            # pext.extra_link_args.extend([])
            extensions.append(pext)
    return extensions
