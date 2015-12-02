

from mako.template import Template

template2d_pyx = Template(filename='template_mako.pyx')
template2d_pxd = Template(filename='template_mako.pxd')

template3d_pyx = Template(filename='template3d_mako.pyx')
template3d_pxd = Template(filename='template3d_mako.pxd')

def make_file(module_name, class_name):

    name_pyx = module_name + '_cy.pyx'
    name_pxd = module_name + '_pxd.pxd'

    if module_name.startswith('fft2d'):
        t_pyx = template2d_pyx
        t_pxd = template2d_pxd
    elif module_name.startswith('fft3d'):
        t_pyx = template3d_pyx
        t_pxd = template3d_pxd

    with open(name_pyx, 'w') as f:
        f.write(t_pyx.render(
            module_name=module_name, class_name=class_name))

    with open(name_pxd, 'w') as f:
        f.write(t_pxd.render(
            module_name=module_name, class_name=class_name))

if __name__ == '__main__':

    variables = (
        ('fft2d_with_fftw1d', 'FFT2DWithFFTW1D'),
        ('fft2d_with_fftw2d', 'FFT2DWithFFTW2D'),
        ('fft2dmpi_with_fftw1d', 'FFT2DMPIWithFFTW1D'),
        ('fft2dmpi_with_fftwmpi2d', 'FFT2DMPIWithFFTWMPI2D'),
        ('fft3d_with_fftw3d', 'FFT3DWithFFTW3D'),
        ('fft3dmpi_with_fftwmpi3d', 'FFT3DMPIWithFFTWMPI3D'),
        ('fft3dmpi_with_pfft', 'FFT3DMPIWithPFFT'),
    )

    for module_name, class_name in variables:
        make_file(module_name, class_name)
