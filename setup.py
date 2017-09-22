
from __future__ import print_function

import sys
import os
from runpy import run_path
from datetime import datetime

from distutils.sysconfig import get_config_var
from setuptools import setup, find_packages

from purepymake import Extension, make_extensions
from config import get_config
from src_cy.make_files_with_mako import make_pyx_files

try:
    from pythran.dist import PythranExtension
    use_pythran = True
except ImportError:
    use_pythran = False


make_pyx_files()

config = get_config()

# Get the long description from the relevant file
with open('README.rst') as f:
    long_description = f.read()
lines = long_description.splitlines(True)
for i, line in enumerate(lines):
    if line.endswith(':alt: Code coverage\n'):
        iline_coverage = i
        break

long_description = ''.join(lines[iline_coverage+2:])

# Get the version from the relevant file
d = run_path('fluidfft/_version.py')
__version__ = d['__version__']

# make a python module from cython files
run_path('src_cy/create_fake_mod_for_doc.py')

src_cpp_dir = 'src_cpp'
src_cy_dir = 'src_cy'
src_base = 'src_cpp/base'
src_cpp_3d = 'src_cpp/3d'
src_cpp_2d = 'src_cpp/2d'


def create_ext(base_name):

    if base_name.startswith('fft2d'):
        dim = '2d'
    elif base_name.startswith('fft3d'):
        dim = '3d'
    else:
        raise ValueError()

    src_cpp_dim = os.path.join(src_cpp_dir, dim)

    source_ends = ['_cy.pyx']
    if base_name.endswith('cufft'):
        source_ends.append('.cu')
    else:
        source_ends.append('.cpp')

    source_files = [base_name + end for end in source_ends]

    base_name = base_name[len('fft2d'):]
    if base_name.startswith('_'):
        base_name = base_name[1:]

    sources = []
    for name_file in source_files:
        if name_file.endswith('_cy.pyx'):
            path = os.path.join(src_cy_dir, name_file)
        else:
            path = os.path.join(src_cpp_dim, name_file)
        sources.append(path)

    sources.extend(
        [os.path.join(src_base, 'base_fft.cpp'),
         os.path.join(src_cpp_dim, 'base_fft' + dim + '.cpp')])
    if base_name.startswith('mpi'):
        sources.extend(
            [os.path.join(src_base, 'base_fftmpi.cpp'),
             os.path.join(src_cpp_dim, 'base_fft' + dim + 'mpi.cpp')])

    return Extension(
        name='fluidfft.fft' + dim + '.' + base_name,
        sources=sources)


base_names = []
if config['fftw']['use']:
        base_names.extend([
            'fft2d_with_fftw1d', 'fft2d_with_fftw2d', 'fft2dmpi_with_fftw1d',
            'fft3d_with_fftw3d'])

if config['fftw-mpi']['use']:
        base_names.extend([
            'fft2dmpi_with_fftwmpi2d', 'fft3dmpi_with_fftwmpi3d'])

if config['cufft']['use']:
    base_names.extend(['fft2d_with_cufft'])
    base_names.extend(['fft3d_with_cufft'])

if config['pfft']['use']:
    base_names.extend(['fft3dmpi_with_pfft'])

if config['p3dfft']['use']:
    base_names.extend(['fft3dmpi_with_p3dfft'])


on_rtd = os.environ.get('READTHEDOCS')
if on_rtd:
    base_names = []
else:
    import mpi4py
    if mpi4py.__version__[0] < '2':
        raise ValueError('Please upgrade to mpi4py >= 2.0')

    import numpy as np

    ext_modules = []
    libraries = set(['fftw3', 'mpi_cxx'])
    lib_dirs = set()
    include_dirs = set(
        [src_cy_dir, src_base, src_cpp_3d, src_cpp_2d,
         'include', mpi4py.get_include(), np.get_include()])

    specials = {}


def update_with_config(key):
    cfg = config[key]
    if len(cfg['dir']) > 0:
        path = cfg['dir']
        include_dirs.add(os.path.join(path, 'include'))
        lib_dirs.add(os.path.join(path, 'lib'))
    elif len(cfg['include_dir']) > 0:
        include_dirs.add(cfg['include_dir'])
    elif len(cfg['library_dir']) > 0:
        lib_dirs.add(cfg['library_dir'])


for base_name in base_names:
    ext_modules.append(create_ext(base_name))
    TMP = os.getenv('FFTW3_INC_DIR')
    if TMP is not None:
        include_dirs.add(TMP)
    TMP = os.getenv('FFTW3_LIB_DIR')
    if TMP is not None:
        lib_dirs.add(TMP)
    if 'fftwmpi' in base_name:
        libraries.add('fftw3_mpi')
        update_with_config('fftw-mpi')
    elif 'pfft' in base_name:
        libraries.update(['fftw3_mpi', 'pfft'])
        update_with_config('pfft')
    elif 'p3dfft' in base_name:
        libraries.add('p3dfft')
        update_with_config('p3dfft')
    elif 'cufft' in base_name:
        libraries.add('cufft')
        update_with_config('cufft')
        specials[''] = {'CC': 'nvcc'}


def modification_date(filename):
    t = os.path.getmtime(filename)
    return datetime.fromtimestamp(t)


def make_pythran_extensions(modules):
    develop = sys.argv[-1] == 'develop'
    extensions = []
    for mod in modules:
        base_file = mod.replace('.', os.path.sep)
        py_file = base_file + '.py'
        # warning: does not work on Windows
        suffix = get_config_var('EXT_SUFFIX') or '.so'
        bin_file = base_file + suffix
        if not develop or not os.path.exists(bin_file) or \
           modification_date(bin_file) < modification_date(py_file):
            pext = PythranExtension(
                mod, [py_file],
                # extra_compile_args=['-O3', '-fopenmp']
            )
            pext.include_dirs.append(np.get_include())
            # bug pythran extension...
            pext.extra_compile_args.extend(['-O3', '-march=native'
                                            # '-fopenmp'
                                        ])
            pext.extra_link_args.extend(['-O3', '-march=native'])
            extensions.append(pext)
    return extensions


if not on_rtd:
    make_extensions(
        ext_modules, include_dirs=include_dirs,
        lib_dirs=lib_dirs, libraries=libraries,
        specials=specials, CC='mpicxx')

if not on_rtd and use_pythran:
    ext_modules = make_pythran_extensions(
        ['fluidfft.fft2d.util_pythran'])
else:
    ext_modules = []


setup(
    name='fluidfft',
    version=__version__,
    description=('Efficient and easy Fast Fourier Transform for Python.'),
    long_description=long_description,
    keywords='Fast Fourier Transform, spectral code',
    author='Pierre Augier',
    author_email='pierre.augier@legi.cnrs.fr',
    url='https://bitbucket.org/fluiddyn/fluidfft',
    license='CeCILL',
    classifiers=[
        # How mature is this project? Common values are
        # 3 - Alpha
        # 4 - Beta
        # 5 - Production/Stable
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        # actually CeCILL License (GPL compatible license for French laws)
        #
        # Specify the Python versions you support here. In particular,
        # ensure that you indicate whether you support Python 2,
        # Python 3 or both.
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Cython',
        'Programming Language :: C'],
    packages=find_packages(exclude=[
        'doc', 'include', 'scripts', 'src_cpp', 'src_cy']),
    install_requires=['fluiddyn'],
    # cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules)
