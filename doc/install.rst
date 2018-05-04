.. _install:

Installation and advice
=======================

Dependencies
------------

- Python 2.7 or >= 3.5

- a C++11 compiler (for example GCC 4.9 or clang)

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

     install/blas_libs

  .. warning::

     In anaconda (or miniconda), Numpy installed with `conda install numpy` is
     built and linked with MKL (an Intel library).  This can be a real plus for
     performance since MKL replaces fftw functions by (usually) faster ones but it
     has a drawback for fft using the library fftw3_mpi (an implementation of
     parallel fft using 1D decomposition by fftw).  MKL implements some fftw
     functions but not all the functions defined in fftw3_mpi. Since the libraries
     are loaded dynamically, if numpy is imported before the fftw_mpi libraries,
     this can lead to very bad issues (segmentation fault, only if numpy is
     imported before the class!). For security, we prefer to automatically disable
     the building of the fft classes using fftw3_mpi when it is detected that
     numpy uses the MKL library where some fftw symbols are defined.

- Cython

- `Pythran <https://github.com/serge-sans-paille/pythran>`_

  We choose to use the new static Python compiler `Pythran
  <https://github.com/serge-sans-paille/pythran>`_ for some functions of the
  operators. Our microbenchmarks show that the performances are as good as what
  we are able to get with Fortran or C++!

  .. warning::

     To reach good performance, we advice to try to put in the file
     `~/.pythranrc` the lines (it seems to work well on Linux, see the `Pythran
     documentation <https://pythonhosted.org/pythran/MANUAL.html>`_):

     .. code:: bash

        [pythran]
        complex_hook = True

  .. warning::

     The compilation of C++ files produced by Pythran can be long and can
     consume a lot of memory. If you encounter any problems, you can try to use
     clang (for example with ``conda install clangdev``) and to enable its use
     in the file `~/.pythranrc` with:

     .. code:: bash

        [compiler]
        CXX=clang++
        CC=clang

- mpi4py (optional, only for mpi runs),

- pyfftw: FluidFFT can of course use pyfftw and it is often a very fast solution
  for undistributed FFT. However, pyfftw is just an optional dependency.

- And of course FFT libraries!

  The libraries are used if they are installed so you shouldn't have any error
  if you build-install FluidFFT without FFT libraries! However, nothing will be
  built and it's not very interesting. So you have to install at least one of
  the supported libraries, let's say at least fftw!

  .. toctree::
     :maxdepth: 1

     install/fft_libs

Environment variables
---------------------

FluidFFT builds its binaries in parallel. It speedups the build process a lot on
most computers. However, it can be a very bad idea on computers with not enough
memory. If you encounter problems, you can force the number of processes used
during the build using the environment variable ``FLUIDDYN_NUM_PROCS_BUILD``::

   export FLUIDDYN_NUM_PROCS_BUILD=2

FluidFFT is also sensible to the environment variable ``FLUIDDYN_DEBUG``::

   export FLUIDDYN_DEBUG=1


Basic installation with pip
---------------------------

If you are in a hurry and that you are not really concerned about performance,
you can use pip directly without any configuration file::

  pip install fluidfft

or::

  pip install fluidfft --user

However, it better to configure the installation of FluidFFT by creating the file
``~/.fluidfft-site.cfg`` and modify it to fit your requirements before the
installation with pip::

  wget https://bitbucket.org/fluiddyn/fluidfft/raw/default/site.cfg.default -O ~/.fluidfft-site.cfg


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

You can also just manually download the package from `the Bitbucket page
<https://bitbucket.org/fluiddyn/fluidfft>`_ or from `the PyPI page
<https://pypi.org/project/fluidfft/>`_.

Configuration file
~~~~~~~~~~~~~~~~~~

For particular installation setup, copy the default configuration file::

  cp site.cfg.default site.cfg

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

   install/occigen
   install/froggy
   install/triolith
   install/beskow
