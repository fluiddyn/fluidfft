.. _install:

Installation and advice
=======================

Installation with pip
---------------------

To install fluidfft, you need a recent Python (>= 3.6) and a C++11 compiler
(for example GCC 4.9 or clang). We explain how to install Python and other
fluidfft dependencies here: `Get a good scientific Python environment
<http://fluiddyn.readthedocs.io/en/latest/get_good_Python_env.html>`_

To install Fluidfft, just run::

  pip install fluidfft

However, fluidfft build is sensible to some options, contained in a
configuration file (``~/.fluidfft-site.cfg`` or ``site.cfg`` in the root
directory) and in environment variables (see below).

Configuration files and FFT libraries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The configuration file contains in particular the list of FFT libraries that
will be used by fluidfft. Here is a list of FFT libraries, with instructions on
how to install them:

.. toctree::
    :maxdepth: 1

    install/fft_libs

The default configuration file can be downloaded with (On some systems,
``wget`` is not installed by default. You may be able to use ``curl``
instead.)::

  wget https://foss.heptapod.net/fluiddyn/fluidfft/raw/branch/default/site.cfg.default -O ~/.fluidfft-site.cfg

Edit one of the configuration files (``~/.fluidfft-site.cfg`` or ``site.cfg``)
as needed.

.. warning::

   By default (without ``~/.fluidfft-site.cfg``), no FFT classes are compiled so
   that fluidfft will only be able to uses its pure-Python FFT classes (using in
   particular pyfftw)!

Environment variables
~~~~~~~~~~~~~~~~~~~~~

The fluidfft build is also sensible to environment variables.

- ``FLUIDDYN_NUM_PROCS_BUILD``

  FluidFFT builds its binaries in parallel. It speedups the build process a lot on
  most computers. However, it can be a very bad idea on computers with not enough
  memory. If you encounter problems, you can force the number of processes used
  during the build using the environment variable ``FLUIDDYN_NUM_PROCS_BUILD``.

- ``FLUIDDYN_DEBUG`` disables parallel build.

- ``DISABLE_PYTHRAN``

  ``DISABLE_PYTHRAN`` disables compilation with Pythran at build time.

- ``FLUIDFFT_TRANSONIC_BACKEND``

  "pythran" by default, it can be set to "python", "numba" or "cython".

- ``FLUIDFFT_DISABLE_MPI`` can be set to disable all MPI libs.

Warning about re-installing fluidfft with new build options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If fluidfft has already been installed and you want to recompile with new
configuration values in ``~/.fluidfft-site.cfg``, you need to really recompile
fluidfft and not just reinstall an already produced wheel. To do this, use::

  pip install fluidfft --no-binary fluidfft -v

``-v`` toggles the verbose mode of pip so that we see the compilation log and
can check that everything goes well.

Install from the repository (still recommended)
-----------------------------------------------

For FluidFFT, we use the revision control software Mercurial and the main
repository is hosted `here <https://foss.heptapod.net/fluiddyn/fluidfft>`_ in
Heptapod. Download the source with something like::

  hg clone https://foss.heptapod.net/fluiddyn/fluidfft

If you are new with Mercurial and Heptapod, you can also read `this short
tutorial
<http://fluiddyn.readthedocs.org/en/latest/mercurial_heptapod.html>`_.

You can create a default configuration file with::

  cp site.cfg.default site.cfg

Edit the configuration file and set environment variables as needed. Build
fluidfft with the command ``make`` which runs::

  pip install -e .[dev]

After the installation, it is a good practice to run the unit tests by running
``make tests`` or ``make tests_mpi``.

Remark on Numpy installed with conda
------------------------------------

In anaconda (or miniconda), Numpy installed with ``conda install numpy`` can be
built and linked with MKL (an Intel library). This can be a real plus for
performance since MKL replaces fftw functions by (usually) faster ones but it
has a drawback for fft using the library fftw3_mpi (an implementation of
parallel fft using 1D decomposition by fftw). MKL implements some fftw
functions but not all the functions defined in fftw3_mpi. Since the libraries
are loaded dynamically, if numpy is imported before the fftw_mpi libraries,
this can lead to very bad issues (segmentation fault, only if numpy is imported
before the class!). For security, we prefer to automatically disable the
building of the fft classes using fftw3_mpi when it is detected that numpy uses
the MKL library where some fftw symbols are defined.

To install with anaconda numpy linked with openblas::

  conda config --add channels conda-forge
  conda install "blas[build=*openblas]" numpy

About using Pythran to compile fluidfft functions
-------------------------------------------------

We choose to use the Python compiler `Pythran
<https://github.com/serge-sans-paille/pythran>`_ for some functions of the
operators. Our microbenchmarks show that the performances are as good as what
we are able to get with Fortran or C++!

.. warning::

  To reach good performance, we advice to try to put in the file
  ``~/.pythranrc`` the lines (it seems to work well on Linux, see the `Pythran
  documentation <https://pythran.readthedocs.io>`_):

  .. code:: bash

    [pythran]
    complex_hook = True

.. warning::

  The compilation of C++ files produced by Pythran can be long and can consume
  a lot of memory. If you encounter any problems, you can try to use clang (for
  example with ``conda install clangdev``) and to enable its use in the file
  ``~/.pythranrc`` with:

  .. code:: bash

    [compiler]
    CXX=clang++
    CC=clang

About mpi4py
------------

If you enable MPI libraries (from the configuration file), ``pip`` will try to
install mpi4py and MPI development files are needed. For example, on Debian
based OS, one can install the package ``libopenmpi-dev``.


Examples of installation
------------------------

.. toctree::
   :maxdepth: 1

   install/occigen
   install/froggy
   install/triolith
   install/beskow
