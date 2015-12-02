
import os

from setuptools import setup, find_packages

from Cython.Distutils import build_ext
from Cython.Distutils.extension import Extension

import mpi4py


# Get the long description from the relevant file
with open('README.rst') as f:
    long_description = f.read()
lines = long_description.splitlines(True)
long_description = ''.join(lines[8:])

# Get the version from the relevant file
from runpy import run_path
d = run_path('fluidfft2d/_version.py')
__version__ = d['__version__']


os.environ["CC"] = 'mpic++'

src_cpp_dir = 'src_cpp'
src_cy_dir = 'src_cy'
src_base = 'src_cpp/base'

include_base = [src_base, 'include', mpi4py.get_include()]
libraries_base = ['fftw3', 'mpi_cxx']


def create_ext(base_name):

    if base_name.startswith('fft2d'):
        dim = '2d'
    elif base_name.startswith('fft3d'):
        dim = '3d'
    else:
        raise ValueError()

    src_cpp_dim = os.path.join(src_cpp_dir, dim)

    source_files = [base_name + '_cy.pyx',
                    base_name + '.cpp']

    base_name = base_name[len('fft2d'):]

    if base_name.startswith('_'):
        base_name = base_name[1:]

    sources = []
    for nfile in source_files:
        if nfile.endswith('_cy.pyx'):
            path = os.path.join(src_cy_dir, nfile)
        else:
            path = os.path.join(src_cpp_dim, nfile)
        sources.append(path)

    sources.extend(
        [os.path.join(src_base, 'base_fft.cpp'),
         os.path.join(src_cpp_dim, 'base_fft' + dim + '.cpp')])
    if base_name.startswith('mpi'):
        sources.extend(
            [os.path.join(src_base, 'base_fftmpi.cpp'),
             os.path.join(src_cpp_dim, 'base_fft' + dim + 'mpi.cpp')])

    include_dirs = include_base + [src_cpp_dim]
    lib_dir = []

    if 'fftwmpi' in base_name:
        libraries = libraries_base + ['fftw3_mpi']
    elif 'pfft' in base_name:
        libraries = libraries_base + ['fftw3_mpi', 'pfft']
        include_dirs += ['/home/users/augier3pi/opt/include']
        lib_dir = ['/home/users/augier3pi/opt/lib']
    else:
        libraries = libraries_base

    return Extension(
        name='fluidfft' + dim + '.' + base_name,
        sources=sources,
        language='c++',
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=lib_dir)


base_names = [
    'fft2d_with_fftw1d', 'fft2d_with_fftw2d', 'fft2dmpi_with_fftw1d',
    'fft2dmpi_with_fftwmpi2d',
    'fft3d_with_fftw3d', 'fft3dmpi_with_fftwmpi3d', 'fft3dmpi_with_pfft']

ext_modules = []
for base_name in base_names:
    ext_modules.append(create_ext(base_name))

setup(
    name='fluidfft',
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
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules)
