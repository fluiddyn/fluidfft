
# actually I had to download it from another computer and copy it with scp...
wget http://www.fftw.org/fftw-3.3.7.tar.gz

tar xf fftw-3.3.7.tar.gz

cd fftw-3.3.7

OPT=$HOME/opt

PREFIX=$OPT/fftw/3.3.7

mkdir -p $PREFIX

./configure CC=icc CFLAGS=-gcc --with-our-malloc16 --enable-threads \
    --enable-openmp --enable-mpi --enable-shared --enable-avx \
    --prefix=$PREFIX

make -j 4 && make install

./configure CC=icc CFLAGS=-gcc --with-our-malloc16 --enable-threads \
    --enable-openmp --enable-mpi --enable-shared --enable-avx \
    --prefix=$PREFIX --enable-float

make -j 4 && make install

./configure CC=icc CFLAGS=-gcc --with-our-malloc16 --enable-threads \
    --enable-openmp --enable-mpi --enable-shared \
    --prefix=$PREFIX --enable-long-double

make -j 4 && make install