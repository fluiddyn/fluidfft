#!/bin/bash

# Customizable variables
# ----------------------
pkgname='fftw'
# FFTW version
pkgver='3.3.6-pl2'
# Directory in which the source tarball will be downloaded and extracted
srcdir=$PWD
# Directory to which the compiled FFTW library will be installed
pkgdir="$HOME/.local/"
export MAKEFLAGS="-j$(nproc)"

# Should be no reason to change anything below
# --------------------------------------------
download() {
  cd ${srcdir}

  if [ ! -f $pkgname-$pkgver.tar.gz ]; then
    wget http://www.fftw.org/$pkgname-$pkgver.tar.gz
  fi
  tar vxzf $pkgname-$pkgver.tar.gz
}

build() {
  cd ${srcdir}

  cp -a ${pkgname}-${pkgver} ${pkgname}-${pkgver}-double
  cp -a ${pkgname}-${pkgver} ${pkgname}-${pkgver}-long-double
  cp -a ${pkgname}-${pkgver} ${pkgname}-${pkgver}-single


  # use upstream default CFLAGS while keeping our -march/-mtune
  CFLAGS+=" -O3 -fomit-frame-pointer -malign-double -fstrict-aliasing -ffast-math"

  CONFIGURE="./configure F77=gfortran CC=gcc CXX=g++ \
	         --prefix=${pkgdir} \
                 --enable-shared \
		 --enable-threads \
		 --enable-openmp \
		 --enable-mpi"

  # build double precision
  cd ${srcdir}/${pkgname}-${pkgver}-double
  $CONFIGURE --enable-sse2 --enable-avx
  make

  # build & install long double precission
  cd ${srcdir}/${pkgname}-${pkgver}-long-double
  $CONFIGURE --enable-long-double
  make

  # build & install single precision
  cd ${srcdir}/${pkgname}-${pkgver}-single
  $CONFIGURE --enable-float --enable-sse --enable-avx
  make
}

package() {
  cd ${srcdir}/${pkgname}-${pkgver}-double
  make install

  cd ${srcdir}/${pkgname}-${pkgver}-long-double
  make install

  cd ${srcdir}/${pkgname}-${pkgver}-single
  make install
}


# Execute the functions above
# ---------------------------
if [ ! -d  ${srcdir}/${pkgname}-${pkgver} ]
then
  download
fi

build
package
