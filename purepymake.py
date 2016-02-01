"""
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
config_vars = sysconfig.get_config_vars()
import subprocess
from copy import copy

short_version = '.'.join([str(i) for i in sys.version_info[:2]])

path_lib_python = os.path.join(sys.prefix, 'lib', 'python' + short_version)
path_include_python = os.path.join(
    sys.prefix, 'include', 'python' + short_version)

path_tmp = 'build/temp.' + '-'.join(
    [platform.system().lower(), platform.machine(), short_version])


def modification_date(filename):
    t = os.path.getmtime(filename)
    return datetime.fromtimestamp(t)


class Extension(object):
    def __init__(self, name, sources=None, language=None):
        self.name = name
        self.sources = sources
        self.language = language


def make_cpp_from_pyx(cpp_file, pyx_file, options=None):
    path_dir = os.path.split(cpp_file)[0]
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    if not os.path.exists(cpp_file) or \
       modification_date(cpp_file) < modification_date(pyx_file):
        command = ['cython', pyx_file, '--cplus', '-o', cpp_file]

        if options is not None and 'include_dirs' in options.keys():
            for inc_dir in options['include_dirs']:
                command += ['-I', inc_dir]
        print(' '.join(command))
        return subprocess.Popen(command)


def make_obj_from_cpp(obj_file, cpp_file, options=None):
    path_dir = os.path.split(obj_file)[0]
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    if not os.path.exists(obj_file) or \
       modification_date(obj_file) < modification_date(cpp_file):

        keys = ['CC', 'OPT', 'BASECFLAGS', 'CONFIGURE_CFLAGS', 'CCSHARED']

        conf_vars = copy(config_vars)

        if options is not None:
            for k, v in options.items():
                conf_vars[k] = v

        command = ' '.join([conf_vars[k] for k in keys])
        command = [w for w in command.split()
                   if w not in ['-g', '-DNDEBUG', '-Wstrict-prototypes']]

        include_dirs = [conf_vars['INCLUDEPY']]
        if options is not None and 'include_dirs' in options.keys():
            include_dirs.extend(options['include_dirs'])

        command += ['-I' + incdir for incdir in include_dirs]
        command += ['-c', cpp_file, '-o', obj_file]
        print(' '.join(command))
        return subprocess.Popen(command)

cxx = config_vars['CXX']
ldshared = config_vars['LDSHARED']

keys = ['CONFIGURE_LDFLAGS']  # , 'CONFIGURE_CFLAGS']

link_command = [w for w in ldshared.split()
                if w not in ['-g']]


def make_ext_from_objs(ext_file, obj_files):

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

        command = link_command + obj_files + ['-o', ext_file]
        print(' '.join(command))
        return subprocess.Popen(command)


def make_extensions(extensions, special=None, **options):

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

    files = {}
    extension_files = ['pyx', 'cpp']

    for ext in extension_files:
        files[ext] = []

    for source in sources:
        for ext in extension_files:
            if source.endswith('.' + ext):
                files[ext].append(source)

    files['o'] = []
    processes = []
    for path in files['pyx']:
        result = os.path.splitext(path)[0] + '.cpp'
        p = make_cpp_from_pyx(result, path, options)
        if p is not None:
            processes.append(p)
        files['cpp'].append(result)

    # wait for cython processes
    while not all([process.poll() is not None for process in processes]):
        sleep(0.1)

    for p in processes:
        if p.returncode != 0:
            raise ValueError()

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
            raise ValueError()

    files['so'] = []

    processes = []
    for ext in extensions:
        result = os.path.join(path_base_output,
                              ext.name.replace('.', os.path.sep) + '.so')
        objects = [
            os.path.join(path_tmp, os.path.splitext(source)[0] + '.o')
            for source in ext.sources]
        p = make_ext_from_objs(result, objects)
        if p is not None:
            processes.append(p)
        files['so'].append(result)

    # wait for linking
    while not all([process.poll() is not None for process in processes]):
        sleep(0.1)

    # print(files)
