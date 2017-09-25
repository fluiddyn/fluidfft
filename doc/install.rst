Installation and advice
=======================

Dependencies
------------

- Python 2.7 or > 3.4

- a C++ compiler

- Cython

- `Pythran <https://github.com/serge-sans-paille/pythran>`_ (optional)

  We choose to use the new static Python compiler Pythran for some functions of
  the operators.

.. warning::

  To reach good performance, we advice to try to put in the file `~/.pythranrc`
  the lines (see the `Pythran documentation
  <https://pythonhosted.org/pythran/MANUAL.html>`_):

  .. code:: bash

     [pythran]
     complex_hook = True
  
- And of course FFT libraries!

The libraries are used if they are installed so you shouldn't have any error if
you build-install FluidFFT without FFT libraries! However, nothing will be
built and it's not very interesting. So you have to install at least one of the
supported libraries, let's say at least fftw!

.. toctree::
   :maxdepth: 1

   fft_libs

Basic installation with pip
---------------------------

If you are in a hurry and that you are not really concerned about performance,
you can use pip::

  pip install fluidfft

or

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
