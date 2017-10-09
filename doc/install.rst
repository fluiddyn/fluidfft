Installation and advice
=======================

Dependencies
------------

- Python 2.7 or >= 3.4

- a C++11 compiler (for example GCC 4.9)

- Numpy

  Make sure to correctly install numpy before anything. 

  .. note::
  
     Be careful, the wheels install with `pip install numpy` can be slow. You
     might get something more efficient by compiling from source using:

     .. code:: bash

        pip install numpy --no-binary numpy
        python -c 'import numpy; numpy.test()'

  .. toctree::
     :maxdepth: 1

     blas_libs

  .. warning::

     In anaconda (or miniconda), Numpy installed with `conda install numpy` is
     built and linked with MKL (an Intel library).  This can be a real plus for
     performance since MKL replaces fftw functions by (usually) faster ones but
     it has a drawback for fft using the library fftw3_mpi (an implementation
     of parallel fft using 1D decomposition by fftw).  MKL implements some fftw
     functions but not all the functions defined in fftw3_mpi. Since the
     libraries are loaded dynamically, if numpy is imported before the fftw_mpi
     libraries, this can lead to very bad issues (segmentation fault, only if
     numpy is imported before the class!). For security, we prefer to
     automatically disable the building of the fft classes using fftw3_mpi when
     it is detected that numpy uses the MKL library where some fftw symbols are
     defined.

- Cython

- `Pythran <https://github.com/serge-sans-paille/pythran>`_

  We choose to use the new static Python compiler `Pythran
  <https://github.com/serge-sans-paille/pythran>`_ for some functions of the
  operators. Our microbenchmarks show that the performances are as good as what
  we are able to get with Fortran or C++!

  .. warning::

     To reach good performance, we advice to try to put in the file
     `~/.pythranrc` the lines (see the `Pythran documentation
     <https://pythonhosted.org/pythran/MANUAL.html>`_):

     .. code:: bash

        [pythran]
        complex_hook = True

- mpi4py (optional, only for mpi runs),
  
- And of course FFT libraries!

  The libraries are used if they are installed so you shouldn't have any error
  if you build-install FluidFFT without FFT libraries! However, nothing will be
  built and it's not very interesting. So you have to install at least one of
  the supported libraries, let's say at least fftw!

  .. toctree::
     :maxdepth: 1

     fft_libs

Basic installation with pip
---------------------------

If you are in a hurry and that you are not really concerned about performance,
you can use pip::

  pip install fluidfft

or::

  pip install fluidfft --user


Install from the repository (recommended)
-----------------------------------------

Get the source code
~~~~~~~~~~~~~~~~~~~

For FluidFFT, we use the revision control software Mercurial and the main
repository is hosted `here <https://bitbucket.org/fluiddyn/fluidfft>`_ in
Bitbucket. Download the source with something like::

  hg clone https://bitbucket.org/fluiddyn/fluidfft

If you are new with Mercurial and Bitbucket, you can also read `this short
tutorial
<http://fluiddyn.readthedocs.org/en/latest/mercurial_bitbucket.html>`_.

If you don't want to use Mercurial, you can also just manually download the
package from `the Bitbucket page <https://bitbucket.org/fluiddyn/fluidfft>`_ or
from `the PyPI page <https://pypi.python.org/pypi/fluidfft>`_.

Configuration file
~~~~~~~~~~~~~~~~~~

For particular installation setup, copy the default configuration file::

  cp site.cfg.example site.cfg

and modify it to fit your requirements.

Build/install
~~~~~~~~~~~~~

Build/install in development mode (with a virtualenv)::

  python setup.py develop

or (without virtualenv)::

  python setup.py develop --user

Of course you can also install FluidDyn with the install command ``python
setup.py install``.

After the installation, it is a good practice to run the unit tests by running
``python -m unittest discover`` from the root directory or from any of the
"test" directories (or just ``make tests`` or ``make tests_mpi``).

Examples of installation
------------------------

.. toctree::
   :maxdepth: 1

   install_occigen
   install_froggy
