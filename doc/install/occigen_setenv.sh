
module purge

export PATH=$HOME/.local/bin:$PATH

# to be able to compile cpp pythran code
module load gcc/6.2.0

# normal environment loading with intel
module load intel/17.2
module load openmpi/intel/2.0.2
module load qt
module load hdf5-seq
module load python/3.6.3
unset PYTHONPATH

# activation the virtualenv (won't work the first time!)
source ~/mypy/bin/activate

OPT=$HOME/opt

export LD_LIBRARY_PATH=$OPT/fftw/3.3.7/lib:$OPT/p3dfft/2.7.5/lib:$OPT/pfft/lib:$LD_LIBRARY_PATH
export CFLAGS=-I$OPT/fftw/3.3.7/include