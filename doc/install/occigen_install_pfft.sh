
OPT=$HOME/opt
ROOTFFTW3=$OPT/fftw/3.3.7
ROOTPFFT=$OPT/pfft

mkdir -p $ROOTPFFT

mkdir -p ~/src
cd ~/src
git clone https://github.com/mpip/pfft
cd pfft
export LANG=C
./bootstrap.sh
./configure --prefix=$ROOTPFFT --with-fftw3=$ROOTFFTW3

make install
