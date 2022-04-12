"""
python setup.py build_ext --inplace

"""

from setuptools import setup
from Cython.Build import cythonize

setup(
    name='grad_cy',
    ext_modules=cythonize('grad_cy.pyx'),
)
