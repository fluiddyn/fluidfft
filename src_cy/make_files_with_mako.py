
import os
from os.path import join
from datetime import datetime

from mako.template import Template

here = os.path.abspath(os.path.dirname(__file__))


def modification_date(filename):
    t = os.path.getmtime(filename)
    return datetime.fromtimestamp(t)


template2d_pyx = Template(filename=join(here, 'template2d_mako.pyx'))
template2d_pxd = Template(filename=join(here, 'template2d_mako.pxd'))

template3d_pyx = Template(filename=join(here, 'template3d_mako.pyx'))
template3d_pxd = Template(filename=join(here, 'template3d_mako.pxd'))


def make_file(module_name, class_name):

    name_pyx = join(here, module_name + '_cy.pyx')
    name_pxd = join(here, module_name + '_pxd.pxd')

    if module_name.startswith('fft2d'):
        t_pyx = template2d_pyx
        t_pxd = template2d_pxd
    elif module_name.startswith('fft3d'):
        t_pyx = template3d_pyx
        t_pxd = template3d_pxd

    if not os.path.exists(name_pyx):
        hastomake = True
    else:
        if modification_date(name_pyx) < modification_date(t_pyx.filename):
            hastomake = True
        else:
            hastomake = False

    if hastomake:
        with open(name_pyx, 'w') as f:
            f.write(t_pyx.render(
                module_name=module_name, class_name=class_name))

    if not os.path.exists(name_pxd):
        hastomake = True
    else:
        if modification_date(name_pxd) < modification_date(t_pxd.filename):
            hastomake = True
        else:
            hastomake = False

    if hastomake:
        with open(name_pxd, 'w') as f:
            f.write(t_pxd.render(
                module_name=module_name, class_name=class_name))


def make_pyx_files():
    variables = (
        ('fft2d_with_fftw1d', 'FFT2DWithFFTW1D'),
        ('fft2d_with_fftw2d', 'FFT2DWithFFTW2D'),
        ('fft2d_with_cufft', 'FFT2DWithCUFFT'),
        ('fft2dmpi_with_fftw1d', 'FFT2DMPIWithFFTW1D'),
        ('fft2dmpi_with_fftwmpi2d', 'FFT2DMPIWithFFTWMPI2D'),
        ('fft3d_with_fftw3d', 'FFT3DWithFFTW3D'),
        ('fft3dmpi_with_fftwmpi3d', 'FFT3DMPIWithFFTWMPI3D'),
        ('fft3dmpi_with_pfft', 'FFT3DMPIWithPFFT'),
        ('fft3dmpi_with_p3dfft', 'FFT3DMPIWithP3DFFT'),
        ('fft3d_with_cufft', 'FFT3DWithCUFFT'),
    )

    for module_name, class_name in variables:
        make_file(module_name, class_name)


if __name__ == '__main__':
    make_pyx_files()
