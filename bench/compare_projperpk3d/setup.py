"""
python setup.py build_ext --inplace

"""

from distutils.core import setup

from pythran.dist import PythranExtension


setup(
    name='proj_setup',
    ext_modules=[PythranExtension('proj_setup', ['proj.py'])],
)
