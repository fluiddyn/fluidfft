#!/bin/bash

# Customizable variables
# ----------------------
pkgname='p3dfft'
# P3DFFT version
pkgver=2.7.6
# Directory in which the source git repository will be downloaded
srcdir="${PWD}"
# Directory to which the compiled p3dfft library will be installed
pkgdir="${HOME}/.local/"

# C and Fortran 90 MPI compilers
CC=mpicc
FC=mpif90

# FFTW
# ----
# You can configure fftwdir by setting an environment variable outside the script
fftwdir=${fftwdir-"${HOME}/.local/"}

# Should be no reason to change anything below
# --------------------------------------------
git_clone() {
  git clone https://github.com/CyrilleBonamy/p3dfft.git ${srcdir}/${pkgname}-${pkgver} --depth=10
  # git clone https://github.com/sdsc/p3dfft.git ${srcdir}/${pkgname}-${pkgver} --depth=10
}

download() {
  mkdir -p ${srcdir}
  cd ${srcdir}

  if [ ! -f ${pkgname}-${pkgver}.tar.gz ]; then
    wget https://github.com/sdsc/p3dfft/archive/v${pkgver}.tar.gz -O ${pkgname}-${pkgver}.tar.gz
  fi
  tar vxzf ${pkgname}-${pkgver}.tar.gz
}

clean() {
  rm -rf ${srcdir}/${pkgname}-${pkgver}
}

build() {
  cd ${srcdir}/${pkgname}-${pkgver}

  libtoolize && aclocal && autoconf && automake --add-missing
  ## If the above fails, use:
  # autoreconf -fvi

  CC=${CC} CCLD=${FC} ./configure \
    --prefix=${pkgdir} \
    --enable-fftw --with-fftw=${fftwdir}

  make
}

package() {
  cd ${srcdir}/${pkgname}-${pkgver}
  make install
  ## If the above fails, use (with caution):
  # make -i install
}


# Execute the functions above
# ---------------------------
clean
if [ ! -d  ${srcdir}/${pkgname}-${pkgver} ]
then
  ## Use any one of the following
  git_clone
  # download
fi
build
package
