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

config_vars = sysconfig.get_config_vars()

try:
    from Cython.Compiler.Main import \
        CompilationOptions, \
        default_options as cython_default_options, \
        compile_single as cython_compile
    can_import_cython = True
except ImportError:
    can_import_cython = False


short_version = '.'.join([str(i) for i in sys.version_info[:2]])

path_lib_python = os.path.join(sys.prefix, 'lib', 'python' + short_version)
# path_include_python = os.path.join(
#     sys.prefix, 'include', 'python' + short_version)

path_tmp = 'build/temp.' + '-'.join(
    [platform.system().lower(), platform.machine(), short_version])


def modification_date(filename):
    t = os.path.getmtime(filename)
    return datetime.fromtimestamp(t)


class CompilationError(Exception):
    def __init__(self, out, err):
        self.message = '\nstdout:\n {}\nstderr:\n{}'.format(out, err)
        super(CompilationError, self). __init__()

    def __str__(self):
        return super(CompilationError, self).__str__() + self.message


class Extension(object):
    def __init__(self, name, sources=None, language=None):
        self.name = name
        self.sources = sources
        self.language = language


def make_cpp_from_pyx(cpp_file, pyx_file, full_module_name=None, options=None):
    path_dir = os.path.split(cpp_file)[0]
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    if not os.path.exists(cpp_file) or \
       modification_date(cpp_file) < modification_date(pyx_file):

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
            compile_time_env=None)
        result = cython_compile(pyx_file, options=options,
                                full_module_name=full_module_name)

        print('cythonize ' + pyx_file)

        return result


def make_obj_from_cpp(obj_file, cpp_file, options=None):
    path_dir = os.path.split(obj_file)[0]
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    if not os.path.exists(obj_file) or \
       modification_date(obj_file) < modification_date(cpp_file):

        keys = ['CC', 'OPT', 'BASECFLAGS', 'CFLAGS', 'CCSHARED']

        conf_vars = copy(config_vars)

        if options is not None:
            for k, v in options.items():
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
                '/opt/cuda/NVIDIA_CUDA-6.0_Samples/common/inc/'
                 ])

        if cpp_file.endswith('.cu'):
            include_dirs.extend([
                '/usr/lib/openmpi/include',
                '/usr/lib/openmpi/include/openmpi'])

        if options is not None and 'include_dirs' in options.keys():
            include_dirs.extend(options['include_dirs'])

        command += ['-I' + incdir for incdir in include_dirs]
        command += ['-c', cpp_file, '-o', obj_file]
        print(' '.join(command))
        return subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def make_ext_from_objs(ext_file, obj_files, lib_dirs=None, libraries=None,
                       specials=None, options=None):

    cond = False
    if not os.path.exists(ext_file):
        cond = True
    else:
        date_ext = modification_date(ext_file)
        if any([date_ext < modification_date(obj_file)
                for obj_file in obj_files]):
            cond = True

    if cond:
        path_dir = os.path.split(ext_file)[0]
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)

        # cxx = config_vars['CXX']
        ldshared = config_vars['LDSHARED']

        # keys = ['CONFIGURE_LDFLAGS']  # , 'CONFIGURE_CFLAGS']

        command = [w for w in ldshared.split()
                   if w not in ['-g']]

        if 'cufft' in ext_file:
            command = 'nvcc -Xcompiler -pthread -shared'.split()

        command += obj_files + ['-o', ext_file]
        if lib_dirs is not None:
            command.extend(['-L' + lib_dir for lib_dir in lib_dirs])

        if libraries is not None:
            command.extend(['-l' + lib for lib in libraries])

        print(' '.join(command))
        return subprocess.Popen(command)


def make_extensions(extensions, lib_dirs=None, libraries=None,
                    special=None, **options):

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
    for pyx_file in files['pyx']:
        cpp_file = os.path.splitext(pyx_file)[0] + '.cpp'

        full_module_name = None
        for ext in extensions:
            if pyx_file in ext.sources:
                full_module_name = ext.name

        result = make_cpp_from_pyx(
            cpp_file, pyx_file,
            full_module_name=full_module_name, options=options)

        files['cpp'].append(cpp_file)

    # compile .cpp files if needed
    processes = []
    for path in files['cpp']:
        result = os.path.join(path_tmp, os.path.splitext(path)[0] + '.o')
        p = make_obj_from_cpp(result, path, options)
        if p is not None:
            processes.append(p)
        files['o'].append(result)

    # wait for compilation
    while not all([process.poll() is not None for process in processes]):
        sleep(0.1)

    for p in processes:
        if p.returncode != 0:
            out, err = p.communicate()
            raise CompilationError(out, err)

    files['so'] = []

    # link .o files to produce the .so files if needed
    processes = []
    for ext in extensions:
        suffix = sysconfig.get_config_var('EXT_SUFFIX') or '.so'
        result = os.path.join(path_base_output,
                              ext.name.replace('.', os.path.sep) + suffix)
        objects = [
            os.path.join(path_tmp, os.path.splitext(source)[0] + '.o')
            for source in ext.sources]
        p = make_ext_from_objs(result, objects,
                               lib_dirs=lib_dirs, libraries=libraries)
        if p is not None:
            processes.append(p)
        files['so'].append(result)

    # wait for linking
    while not all([process.poll() is not None for process in processes]):
        sleep(0.1)

    # print(files)
