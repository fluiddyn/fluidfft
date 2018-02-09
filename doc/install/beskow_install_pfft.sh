#!/bin/bash

# Customizable variables
# ----------------------
pkgname='pfft'
# PFFT version
pkgver="1.0.8-alpha"
# Directory in which the source git repository will be downloaded
srcdir="${PWD}"
# Directory to which the compiled pfft library will be installed
pkgdir="/cfs/klemming/nobackup/${USER:0:1}/${USER}/opt/pkg/${pkgname}-${pkgver}"
export MAKEFLAGS="-j$(nproc)"

# C and Fortran 90 MPI compilers
export CC=mpiicc
export FC=mpiifort
export CFLAGS="-xHost"
export LDFLAGS="-nofor-main"
# CC="cc"
# FC="ftn"

# FFTW
# ----
# FFTW == 3.3.4 requires patching, whereas 3.3.5 andlater versions should work
# as it is. See: https://github.com/mpip/pfft#install

# You can use the same fftw directory that for p3dfft
# fftwdir="/opt/fftw/3.3.4.0/haswell"
fftwdir="/cfs/klemming/nobackup/${USER:0:1}/${USER}/opt"

# Alternatively, set fftwdir as an empty string and mention fftw include and
# library directories seperately below
fftwinc=""
fftwlib=""

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
  rm -rf ${pkgdir}
}

build() {
  cd ${srcdir}/${pkgname}-${pkgver}
  export LANG=C
  ./bootstrap.sh
  CONFIGURE="./configure \
            --prefix=${pkgdir} \
            CC=${CC} FC=${FC} MPICC=${CC} MPIFC=${FC} "
            # --host=x86_64-unknown-linux-gnu
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
  set -e
  cd ${srcdir}/${pkgname}-${pkgver}
  make install

  set +e
  cd ${pkgdir}/..
  stow -v $pkgname-$pkgver
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
