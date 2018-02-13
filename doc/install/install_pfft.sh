#!/bin/bash

# Customizable variables
# ----------------------
pkgname='pfft'
# PFFT version
pkgver="1.0.8-alpha"
# Directory in which the source git repository will be downloaded
srcdir="${PWD}"
# Directory to which the compiled pfft library will be installed
pkgdir="${HOME}/.local/"
export MAKEFLAGS="-j$(nproc)"

# C and Fortran 90 MPI compilers
CC=mpicc
FC=mpif90

# FFTW
# ----
# FFTW == 3.3.4 requires patching, whereas 3.3.5 and later versions should work
# as it is. See: https://github.com/mpip/pfft#install

# You can configure fftwdir by setting an environment variable outside the script
fftwdir=${fftwdir-"${HOME}/.local/"}

# Alternatively, set fftwdir as an empty string and set fftwinc and fftwlib
fftwinc=${fftwinc-""}
fftwlib=${fftwlib-""}

# Should be no reason to change anything below
# --------------------------------------------
git_clone() {
  git clone https://github.com/mpip/pfft.git ${srcdir}/${pkgname}-${pkgver} --depth=10
}

download() {
  mkdir -p ${srcdir}
  cd ${srcdir}

  if [ ! -f ${pkgname}-${pkgver}.tar.gz ]; then
    wget http://www.tu-chemnitz.de/~potts/workgroup/pippig/software/${pkgname}-${pkgver}.tar.gz
  fi
  tar vxzf ${pkgname}-${pkgver}.tar.gz
}

clean() {
  rm -rf ${srcdir}/${pkgname}-${pkgver}
}

build() {
  cd ${srcdir}/${pkgname}-${pkgver}
  export LANG=C
  ./bootstrap.sh
  CONFIGURE="./configure \
            --prefix=${pkgdir} \
            CC=${CC} FC=${FC} MPICC=${CC} MPIFC=${FC} "
  if [ -n "$fftwdir" ]; then
    CONFIGURE+="--with-fftw3=${fftwdir}"
  else
    CONFIGURE+="CPPFLAGS=-I${fftwinc}  LDFLAGS=-L${fftwlib}"
  fi
  echo ${CONFIGURE}
  ${CONFIGURE}
  make
}

package() {
  cd ${srcdir}/${pkgname}-${pkgver}
  make install
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
