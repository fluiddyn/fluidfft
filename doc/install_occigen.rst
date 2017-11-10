Installation on occigen
=======================

First install mercurial in python 2 because mercurial still works
better in python 2 and because there is no pip in the system python 2

.. code-block:: bash

   module load intel
   module load python/2.7.13
   pip2 install mercurial --user

The commands to load the environment:

.. code-block:: bash

   # normal environment loading with intel
   export PATH=$HOME/.local/bin:$PATH
   module load intel/17.2
   module load openmpi/intel/2.0.2
   module load qt
   module load hdf5-seq
   module load python/3.6.3
   unset PYTHONPATH

   # to be able to compile cpp pythran code
   module load gcc/6.2.0

   # activation the virtualenv (won't work the first time!)
   source ~/mypy/bin/activate

Prepare the environment:

.. code-block:: bash

   # creation of a virtualenv
   virtualenv --system-site-packages mypy
   source ~/mypy/bin/activate

   pip install mpi4py --no-deps --no-binary mpi4py
   pip install h5py --no-deps --no-binary h5py

   pip install pythran colorlog
   pip install fluiddyn
