set -e

pip install pip virtualenv -U

CWD=$(pwd)
BASE=/tmp/tmp_debug

rm -rf $BASE
mkdir $BASE
cd $BASE
virtualenv myvenv
. myvenv/bin/activate
pip install numpy cython pytest

git clone https://bitbucket.org/mpi4py/mpi4py-fft.git
cd mpi4py-fft
git checkout 67dfed980115108c76abb7e865860b5da98674f9
python setup.py develop

cd $BASE
hg clone ssh://hg@foss.heptapod.net/fluiddyn/fluidfft
cd fluidfft
hg up a50bbec26559
cp $CWD/site.cfg.debug site.cfg
python setup.py develop

cd $BASE
hg clone https://github.com/pyFFTW/pyFFTW.git
