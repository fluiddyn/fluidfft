Installation on occigen
=======================

First install mercurial in python 2 because mercurial still works
better in python 2 and because there is no pip in the system python 2

.. code-block:: bash

   module load intel
   module load python/2.7.13
   pip2 install mercurial --user

We write in a file occigen_setenv.sh the commands to setup the environment:

.. literalinclude:: occigen_setenv.sh
   :language: shell

We source this file::

  source occigen_setenv.sh

We can then prepare the python environment:

.. code-block:: bash

   # creation of a virtualenv
   virtualenv --system-site-packages mypy
   source ~/mypy/bin/activate

   pip install mpi4py --no-deps --no-binary mpi4py
   pip install h5py --no-deps --no-binary h5py

   pip install pythran colorlog
   pip install fluiddyn

Finally, we can install fluidfft::

  hg clone http://bitbucket.org/fluiddyn/fluidfft
  cd fluidfft
  make develop
