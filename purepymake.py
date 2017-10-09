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

from __future__ import print_function

import sys
import platform
from time import sleep
import os
from datetime import datetime
from distutils import sysconfig
import subprocess
from copy import copy
import importlib
import multiprocessing

config_vars = sysconfig.get_config_vars()

try:
    from Cython.Compiler.Main import \
        CompilationOptions, \
        default_options as cython_default_options, \
        compile_single as cython_compile
    can_import_cython = True
except ImportError:
    can_import_cython = False

try:
    import mpi4py
    can_import_mpi4py = True
except ImportError:
    can_import_mpi4py = False

short_version = '.'.join([str(i) for i in sys.version_info[:2]])

path_lib_python = os.path.join(sys.prefix, 'lib', 'python' + short_version)

path_tmp = 'build/temp.' + '-'.join(
    [platform.system().lower(), platform.machine(), short_version])


def can_import(pkg):
    try:
        importlib.import_module(pkg)
    except ImportError:
        return False
    else:
        return True


def check_deps():
    def check_and_print(pkg='', result=None):
        if result is None:
            result = can_import(pkg)

        print('{} installed: '.format(pkg).rjust(25) + repr(result))

    print('*' * 40)
    check_and_print('numpy')
    check_and_print('mpi4py', can_import_mpi4py)
    check_and_print('cython', can_import_cython)
    check_and_print('pythran')
    check_and_print('mako')
    print('*' * 40)


check_deps()


def modification_date(filename):
    t = os.path.getmtime(filename)
    return datetime.fromtimestamp(t)


class CompilationError(Exception):
    def __init__(self, command, out, err):
        self.message = '\ncommand:\n{}\nstdout:\n {}\nstderr:\n{}'.format(
            command, out.decode(), err.decode())
        super(CompilationError, self). __init__()

    def __str__(self):
        return super(CompilationError, self).__str__() + self.message


class Extension(object):
    def __init__(self, name, sources=None, language=None, ):
        self.name = name
        self.sources = sources
        self.language = language


def has_to_build(output_file, input_files):
    if not os.path.exists(output_file):
        return True
    mod_date_output = modification_date(output_file)
    for input_file in input_files:
        if mod_date_output < modification_date(input_file):
            return True
    return False


class CommandsRunner(object):
    nb_proc = 4

    def __init__(self, commands):
        self.commands = commands
        self._processes = []

    def run(self):
        while len(self.commands) != 0:
            if len(self._processes) < self.nb_proc:
                command = self.commands.pop()
                self._launch_process(command)

            sleep(0.1)
            self._check_processes()

        while len(self._processes) != 0:
            sleep(0.1)
            self._check_processes()

    def _launch_process(self, command):
        print('launching command:\n' + ' '.join(command))
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        process.command = ' '.join(command)
        self._processes.append(process)

    def _check_processes(self):
        for process in self._processes.copy():
            if process.poll() is not None:
                self._processes.remove(process)
                if process.returncode != 0:
                    out, err = process.communicate()
                    raise CompilationError(process.command, out, err)


class FunctionsRunner(CommandsRunner):
    def _launch_process(self, command):
        print('launching command:\n', command)
        func, args, kwargs = command
        process = multiprocessing.Process(
            target=func, args=args, kwargs=kwargs)
        process.start()
        process.command = command
        self._processes.append(process)

    def _check_processes(self):
        for process in self._processes.copy():
            if process.exitcode is not None:
                self._processes.remove(process)


def make_function_cpp_from_pyx(cpp_file, pyx_file,
                               full_module_name=None, options=None):
    path_dir = os.path.split(cpp_file)[0]
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)

    if not has_to_build(cpp_file, (pyx_file,)):
        return

    if not can_import_cython:
        raise ImportError('Can not import Cython.')

    include_path = None
    if options is not None and 'include_dirs' in options.keys():
        include_path = list(options['include_dirs'])

    options = CompilationOptions(
        cython_default_options,
        include_path=include_path,
        output_file=cpp_file,
        cplus=True,
        compile_time_env={'MPI4PY': can_import_mpi4py})

    # return (func, args, kwargs)
    return (cython_compile, (pyx_file,),
            {'options': options, 'full_module_name': full_module_name})


def make_command_obj_from_cpp(obj_file, cpp_file, options=None):
    path_dir = os.path.split(obj_file)[0]
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)

    if not has_to_build(obj_file, (cpp_file,)):
        return

    keys = ['CXX', 'OPT', 'BASECFLAGS', 'CFLAGS', 'CCSHARED']

    conf_vars = copy(config_vars)

    for k in keys:
        print(k, conf_vars[k])

    # problem: it replaces the value (add it better?)
    if options is not None:
        for k, v in options.items():
            if v is not None:
                conf_vars[k] = v

    command = ' '.join([conf_vars[k] for k in keys])

    if cpp_file.endswith('.cu'):
        command = (
            'nvcc -m64 '
            '-gencode arch=compute_20,code=sm_20 '
            '-gencode arch=compute_30,code=sm_30 '
            '-gencode arch=compute_32,code=sm_32 '
            '-gencode arch=compute_35,code=sm_35 '
            '-gencode arch=compute_50,code=sm_50 '
            '-gencode arch=compute_50,code=compute_50 -Xcompiler -fPIC')

    command = [w for w in command.split()
               if w not in ['-g', '-DNDEBUG', '-Wstrict-prototypes']]

    include_dirs = [conf_vars['INCLUDEPY']]

    if 'cufft' in cpp_file:
        include_dirs.extend([
            '/opt/cuda/NVIDIA_CUDA-6.0_Samples/common/inc/'])

    if cpp_file.endswith('.cu'):
        include_dirs.extend([
            '/usr/lib/openmpi/include',
            '/usr/lib/openmpi/include/openmpi'])

    if options is not None and 'include_dirs' in options.keys():
        include_dirs.extend(options['include_dirs'])

    command += ['-I' + incdir for incdir in include_dirs]
    command += ['-c', cpp_file, '-o', obj_file]
    return command


def make_command_ext_from_objs(
        ext_file, obj_files, lib_dirs=None,
        libraries=None, options=None):

    if not has_to_build(ext_file, obj_files):
        return

    path_dir = os.path.split(ext_file)[0]
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)

    cxx = config_vars['CXX']
    ldshared = config_vars['LDSHARED']

    command = [w for w in ldshared.split()
               if w not in ['-g']]

    if can_import_mpi4py:
        command[0] = os.getenv('MPICXX', 'mpicxx')
    else:
        command[0] = cxx.split()[0]

    if 'cufft' in ext_file:
        command = 'nvcc -Xcompiler -pthread -shared'.split()

    command += obj_files + ['-o', ext_file]
    if lib_dirs is not None:
        command.extend(['-L' + lib_dir for lib_dir in lib_dirs])

    if libraries is not None:
        command.extend(['-l' + lib for lib in libraries])

    return command


def make_extensions(extensions, lib_dirs=None, libraries=None,
                    **options):

    if all(command not in sys.argv for command in [
            'build_ext', 'install', 'develop']):
        return

    if '--inplace' in sys.argv or 'develop' in sys.argv:
        path_base_output = '.'
    else:
        path_base_output = path_tmp

    sources = set()
    for ext in extensions:
        sources.update(ext.sources)

    # prepare a dictionary listing all files
    files = {}
    extension_files = ['pyx', 'cpp']

    for ext in extension_files:
        files[ext] = []

    for source in sources:
        for ext in extension_files:
            if source.endswith('.' + ext):
                files[ext].append(source)
        if source.endswith('.cu'):
            files['cpp'].append(source)
    files['o'] = []

    # cythonize .pyx files if needed
    commands = []
    for pyx_file in files['pyx']:
        cpp_file = os.path.splitext(pyx_file)[0] + '.cpp'

        full_module_name = None
        for ext in extensions:
            if pyx_file in ext.sources:
                full_module_name = ext.name

        command = make_function_cpp_from_pyx(
            cpp_file, pyx_file,
            full_module_name=full_module_name, options=options)
        if command is not None:
            commands.append(command)

        files['cpp'].append(cpp_file)

    FunctionsRunner(commands).run()

    # compile .cpp files if needed
    commands = []
    for path in files['cpp']:
        result = os.path.join(path_tmp, os.path.splitext(path)[0] + '.o')
        command = make_command_obj_from_cpp(result, path, options)
        if command is not None:
            commands.append(command)
        files['o'].append(result)

    CommandsRunner(commands).run()

    # link .o files to produce the .so files if needed
    files['so'] = []
    commands = []
    for ext in extensions:
        suffix = sysconfig.get_config_var('EXT_SUFFIX') or '.so'
        result = os.path.join(path_base_output,
                              ext.name.replace('.', os.path.sep) + suffix)
        objects = [
            os.path.join(path_tmp, os.path.splitext(source)[0] + '.o')
            for source in ext.sources]
        command = make_command_ext_from_objs(
            result, objects, lib_dirs=lib_dirs, libraries=libraries)
        if command is not None:
            commands.append(command)
        files['so'].append(result)

    # wait for linking
    CommandsRunner(commands).run()
