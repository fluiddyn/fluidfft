Installation on Triolith
========================

First install mercurial in python 2.

.. code-block:: bash

   module load gcc/6.2.0
   module load python/2.7.12
   pip2 install mercurial --user

Load necessary modules

.. code-block:: bash

   module load python3/3.6.1
   module load openmpi/1.6.5-g44
   module load hdf5/1.8.11-i1214-parallel
   module load libtool/2.4
   module load autoconf/2.69

To build and install in FFTW, P3DFFT and PFFT, use the scripts provided in this
page:

  .. toctree::
     :maxdepth: 1

     fft_libs

but with few modifications:

 - P3DFFT

   - Use ``autoreconf -fvi`` just before ``./configure ...`` and ``make``.
   - Use ``make -i install`` to finish installation, while making a note of
     the errors encountered. Ignore if only P3DFFT samples fail to build.

Setup a virtual environment, using ``virtualenv``, or ``mkvirtualenv`` command
from ``virtualenvwrapper`` package or simply using Python's built-in module
``python -m venv`` module.

Set up ``~/.pythranrc`` and ``~/.fluidfft-site.cfg`` and install python packages
as described in occigen installation:

  .. toctree::
     :maxdepth: 1

     occigen_install
