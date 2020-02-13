.. warning ::

   Our repositories in Bitbucket.org will soon be deleted! Our new home:
   https://foss.heptapod.net/fluiddyn (`more details
   <https://fluiddyn.readthedocs.io/en/latest/advice_developers.html>`_).

======
|logo|
======

*efficient and easy Fast Fourier Transform for Python*

|release| |pyversions| |docs| |coverage| |travis|

.. |logo| image:: https://foss.heptapod.net/fluiddyn/fluidfft/raw/default/doc/logo.png
   :alt: FluidFFT

.. |release| image:: https://img.shields.io/pypi/v/fluidfft.svg
   :target: https://pypi.org/project/fluidfft/
   :alt: Latest version

.. |pyversions| image:: https://img.shields.io/pypi/pyversions/fluidfft.svg
   :alt: Supported Python versions

.. |docs| image:: https://readthedocs.org/projects/fluidfft/badge/?version=latest
   :target: http://fluidfft.readthedocs.org
   :alt: Documentation status

.. |coverage| image:: https://codecov.io/gh/fluiddyn/fluidfft/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/fluiddyn/fluidfft
   :alt: Code coverage

.. |travis| image:: https://travis-ci.org/fluiddyn/fluidfft.svg?branch=master
    :target: https://travis-ci.org/fluiddyn/fluidfft

.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/s75as4ikeetrk6f3/branch/default?svg=true
   :target: https://ci.appveyor.com/project/fluiddyn/fluiddyn
   :alt: AppVeyor status

.. |binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/fluiddyn/fluidfft/master?urlpath=lab/tree/doc/ipynb
   :alt: Binder notebook

Fluidfft provides C++ classes and their Python wrapper classes written in
Cython useful to perform Fast Fourier Transform (FFT) with different libraries,
in particular

- `fftw3 <http://www.fftw.org/>`_ and `fftw3-mpi
  <http://www.fftw.org/fftw3_doc/Distributed_002dmemory-FFTW-with-MPI.html>`_

- `pfft <https://github.com/mpip/pfft>`_

- `p3dfft <https://github.com/sdsc/p3dfft>`_

- `mpi4py-fft <https://bitbucket.org/mpi4py/mpi4py-fft>`_

- `cufft <https://developer.nvidia.com/cufft>`_ (fft library by CUDA
  running on GPU)

`pfft <https://github.com/mpip/pfft>`_, `p3dfft
<https://github.com/sdsc/p3dfft>`_ and `mpi4py-fft
<https://bitbucket.org/mpi4py/mpi4py-fft>`_ are specialized in computing FFT
efficiently on several cores of big clusters. The data can be split in pencils
and can be distributed on several processes.

**Documentation**: https://fluidfft.readthedocs.io

Getting started
---------------
To try fluidfft without installation: |binder|

For a **basic installation** which relies only on a ``pyFFTW`` interface; or
provided you have the optional FFT libaries, that you need, installed and
discoverable in your path (see environment variables ``LIBRARY_PATH``,
``LD_LIBRARY_PATH``, ``CPATH``) it should be sufficient to run::

  pip install fluidfft [--user]

Add ``--user`` flag if you are installing without setting up a virtual
environment.

Installation
------------

To take full advantage of fluidfft, consider installing the following
(optional) dependencies and configurations before installing fluidfft. Click on
the links to know more:

1. OpenMPI or equivalent
2. FFT libraries such as MPI-enabled FFTW (for 2D and 3D solvers) and P3DFFT,
   PFFT (for 3D solvers) either using a package manager or `from source
   <https://fluidfft.readthedocs.io/en/latest/install/fft_libs.html>`__
3. Python packages ``fluiddyn mako cython pyfftw pythran mpi4py``
4. `A C++11 compiler and BLAS
   libraries <https://github.com/serge-sans-paille/pythran#installation>`__ and
   `configure
   <https://fluidfft.readthedocs.io/en/latest/install.html#dependencies>`__
   ``~/.pythranrc`` to customize compilation of Pythran extensions
5. `Configure
   <https://fluidfft.readthedocs.io/en/latest/install.html#basic-installation-with-pip>`__
   ``~/.fluidfft-site.cfg`` to detect the FFT libraries and install
   ``fluidfft``

**Note**: Detailed instructions to install the above dependencies using Anaconda
/ Miniconda or in a specific operating system such as Ubuntu, macOS etc. can be
found `here
<https://fluiddyn.readthedocs.io/en/latest/get_good_Python_env.html>`__.

C++ API
*******

See a `working minimal example with Makefile
<https://fluidfft.readthedocs.io/en/latest/examples/cpp.html>`__  which
illustrates how to use the C++ API.

Tests
-----

From the root directory::

  make tests
  make tests_mpi

Or, from the root directory or any of the "test" directories::

  pytest -s
  mpirun -np 2 pytest -s


How does it work?
-----------------

Fluidfft provides classes to use in a transparent way all these libraries with
an unified API. These classes are not limited to just performing Fourier
transforms. They are also an elegant solution to efficiently perform operations
on data in real and spectral spaces (gradient, divergence, rotational, sum over
wavenumbers, computation of spectra, etc.) and easily deal with the data
distribution (gather the data on one process, scatter the data to many
processes) without having to know the internal organization of every FFT
library.

Fluidfft hides the internal complication of (distributed) FFT libraries and
allows the user to find (by benchmarking) and to choose the most efficient
solution for a particular case. Fluidfft is therefore a very useful tool to
write HPC applications using FFT, as for example pseudo-spectral simulation
codes. In particular, fluidfft is used in the Computational Fluid Dynamics
(CFD) framework `fluidsim <http://fluidsim.readthedocs.org>`_.

License
-------

Fluidfft is distributed under the CeCILL_ License, a GPL compatible
french license.

.. _CeCILL: http://www.cecill.info/index.en.html

Metapapers and citations
------------------------

If you use FluidFFT to produce scientific articles, please cite our metapapers
presenting the `FluidDyn project
<https://openresearchsoftware.metajnl.com/articles/10.5334/jors.237/>`__
and `Fluidfft
<https://openresearchsoftware.metajnl.com/articles/10.5334/jors.238/>`__:

.. code ::

    @article{fluiddyn,
    doi = {10.5334/jors.237},
    year = {2019},
    publisher = {Ubiquity Press,  Ltd.},
    volume = {7},
    author = {Pierre Augier and Ashwin Vishnu Mohanan and Cyrille Bonamy},
    title = {{FluidDyn}: A Python Open-Source Framework for Research and Teaching in Fluid Dynamics
        by Simulations,  Experiments and Data Processing},
    journal = {Journal of Open Research Software}
    }

    @article{fluidfft,
    doi = {10.5334/jors.238},
    year = {2019},
    publisher = {Ubiquity Press,  Ltd.},
    volume = {7},
    author = {Ashwin Vishnu Mohanan and Cyrille Bonamy and Pierre Augier},
    title = {{FluidFFT}: Common {API} (C$\mathplus\mathplus$ and Python)
        for Fast Fourier Transform {HPC} Libraries},
    journal = {Journal of Open Research Software}
    }
