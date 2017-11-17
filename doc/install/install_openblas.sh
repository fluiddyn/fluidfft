#!/bin/bash

# Customizable variables
# ----------------------
pkgname='OpenBLAS'
pkgver=0.2.20
_lapackver=3.7.0
# Directory in which the source tarball will be downloaded and extracted
srcdir=$PWD
# Directory to which the compiled OpenBLAS library will be installed
pkgdir="$HOME/.local/${pkgname}-${pkgver}"
fcompiler="gfortran"

# Should be no reason to change anything below
# --------------------------------------------
_config="FC=${fcompiler} USE_OPENMP=0 USE_THREAD=1 \
          NO_LAPACK=0 BUILD_LAPACK_DEPRECATED=1\
          MAKE_NB_JOBS=$(nproc)
        "

download() {
  mkdir -p ${srcdir}
  cd ${srcdir}

  if [ ! -f ${pkgname}-${pkgver}.tar.gz ]; then
    wget http://github.com/xianyi/OpenBLAS/archive/v${pkgver}.tar.gz -O ${pkgname}-${pkgver}.tar.gz
  fi
  tar vxzf ${pkgname}-${pkgver}.tar.gz
}

build(){
  cd ${srcdir}/${pkgname}-${pkgver}

  make ${_config} libs netlib shared
}

check(){
  cd ${srcdir}/${pkgname}-${pkgver}

  make ${_config} tests
}

package(){
  cd ${srcdir}/${pkgname}-${pkgver}

  make ${_config} PREFIX=${pkgdir} install
  cd ${pkgdir}
  # BLAS
  ln -sf libopenblas.so libblas.so
  ln -sf libopenblas.so libblas.so.${_lapackver:0:1}
  ln -sf libopenblas.so libblas.so.${_lapackver}
  # CBLAS
  ln -sf libopenblas.so libcblas.so
  ln -sf libopenblas.so libcblas.so.${_lapackver:0:1}
  ln -sf libopenblas.so libcblas.so.${_lapackver}
  # LAPACK
  ln -sf libopenblas.so liblapack.so
  ln -sf libopenblas.so liblapack.so.${_lapackver:0:1}
  ln -sf libopenblas.so liblapack.so.${_lapackver}
  # LAPACKE
  ln -sf libopenblas.so liblapacke.so
  ln -sf libopenblas.so liblapacke.so.${_lapackver:0:1}
  ln -sf libopenblas.so liblapacke.so.${_lapackver}

  echo "
Installation complete!

Create $HOME/.numpy-site.cfg with

[openblas]
libraries = openblas
library_dirs = ${pkgdir}/lib
include_dirs = ${pkgdir}/include
runtime_library_dirs = ${pkgdir}/lib

to use this OpenBLAS installation.
"
}

# Execute the functions above
# ---------------------------
if [ ! -d  ${srcdir}/${pkgname}-${pkgver} ]
then
  download
fi

build
check
package
