
from __future__ import print_function

import os
from runpy import run_path

from setuptools import setup, find_packages

from purepymake import Extension, make_extensions
from config import get_config
from src_cy.make_files_with_mako import make_pyx_files

make_pyx_files()

config = get_config()

# Get the long description from the relevant file
with open('README.rst') as f:
    long_description = f.read()
lines = long_description.splitlines(True)
long_description = ''.join(lines[12:])

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
    elif 'pfft' in base_name:
        libraries.update(['fftw3_mpi', 'pfft'])
#        include_dirs.update(['/opt/pfft/1.0.6/include'])
        lib_dirs.update(['/opt/pfft/1.0.6/lib'])
    elif 'p3dfft' in base_name:
        libraries.update(['p3dfft'])
#        include_dirs.update(['/opt/p3dfft/2.7.4-mt/include'])
#        lib_dirs.update(['/opt/p3dfft/2.7.4-mt/lib'])
    elif 'cufft' in base_name:
        libraries.add('cufft')
        include_dirs.update(['/opt/cuda/7.5/include'])
        lib_dirs.update(['/opt/cuda/7.5/lib64'])
        specials[''] = {'CC': 'nvcc'}


if not on_rtd:
    make_extensions(
        ext_modules, include_dirs=include_dirs,
        lib_dirs=lib_dirs, libraries=libraries,
        specials=specials, CC='mpicxx')


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
        'Development Status :: 3 - Alpha',
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
        # 'Programming Language :: Python :: 3',
        # 'Programming Language :: Python :: 3.3',
        # 'Programming Language :: Python :: 3.4',
        'Programming Language :: Cython',
        'Programming Language :: C'],
    packages=find_packages(exclude=[
        'doc', 'include', 'scripts', 'src_cpp', 'src_cy']),
    install_requires=['fluiddyn', 'mako']
    # cmdclass={'build_ext': build_ext},
    # ext_modules=ext_modules
)
