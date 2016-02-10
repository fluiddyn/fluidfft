Installation and advice
=======================


Dependencies
------------

- Python 2.7

- a C++ compiler

- Cython
  
- FFT libraries:

  The libraries are used if they are installed so you shouldn't have any error
  if you build-install FluidFFT without FFT libraries! However, nothing will be
  built and it's not very interesting. So you have to install at least one
  library among the following ones:

  * fftw (better to build with OpenMP enabled)

  .. todo::

     Use OpenMP...

    
  * fftw-mpi

  * pfft

  * p3dfft

.. toctree::
   :maxdepth: 1

   dependencies
    
.. todo::

   Add a page on building of FFT libraries.

Note that there are wrappers to use the MKL library (FFT by intel) using the FFTW
  API.

  .. todo::
     Try to use MKL wrappers.


Install in development mode (recommended)
-----------------------------------------

FluidFFT is still in alpha version ("testing for developers"!).  So I would
advice to work "as a developer", i.e. to get the source code and to use
revision control and the development mode of the Python installer.

For FluidFFT, I use the revision control software Mercurial and the main
repository is hosted `here <https://bitbucket.org/fluiddyn/fluidfft>`_ in
Bitbucket. I would advice to fork this repository (click on "Fork") and to
clone your newly created repository to get the code on your computer (click on
"Clone" and run the command that will be given). If you are new with Mercurial
and Bitbucket, you can also read `this short tutorial
<http://fluiddyn.readthedocs.org/en/latest/mercurial_bitbucket.html>`_.

If you really don't want to use Mercurial, you can also just manually
download the package from `the Bitbucket page
<https://bitbucket.org/fluiddyn/fluidfft>`_ or from `the PyPI page
<https://pypi.python.org/pypi/fluidfft>`_.

To install in development mode (with a virtualenv)::

  python setup.py develop

or (without virtualenv)::

  python setup.py develop --user

Of course you can also install FluidDyn with the install command ``python
setup.py install``.

After the installation, it is a good practice to run the unit tests by
running ``python -m unittest discover`` from the root directory or
from any of the "test" directories (or just ``make tests``).

Installation with pip
---------------------

FluidFFT can also be installed from the Python Package Index::

  pip install fluidfft --pre

The ``--pre`` option of pip allows the installation of a pre-release version.

However, the project is so new that it is better to have the last version (from
the mercurial repository hosted on Bitbucket).
