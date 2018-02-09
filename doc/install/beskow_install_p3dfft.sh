#!/bin/bash

# Customizable variables
# ----------------------
pkgname='p3dfft'
# P3DFFT version
pkgver=2.7.6
# Directory in which the source git repository will be downloaded
srcdir="${PWD}"
# Directory to which the compiled p3dfft library will be installed
pkgdir="/cfs/klemming/nobackup/${USER:0:1}/${USER}/opt/pkg/${pkgname}-${pkgver}"

# C and Fortran 90 MPI compilers
export CC="mpiicc"
export FC="mpiifort"
export CFLAGS="-xHost"
export LDFLAGS="-nofor-main"
# CC="cc"
# FC="ftn"

# FFTW
# ----
# fftwdir="/opt/fftw/3.3.4.0/haswell"
fftwdir="/cfs/klemming/nobackup/${USER:0:1}/${USER}/opt"
autotoolsdir="$fftwdir"

# Should be no reason to change anything below
# --------------------------------------------
git_clone() {
  git clone https://github.com/CyrilleBonamy/p3dfft.git ${srcdir}/${pkgname}-${pkgver} --depth=1
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
  rm -rf ${pkgdir}
}

prepare() {
  cd ${srcdir}/${pkgname}-${pkgver}
  # Assuming you had installed and stowed libtool into $autotoolsdir
  cat "${autotoolsdir}/share/aclocal/libtool.m4" \
      "${autotoolsdir}/share/aclocal/ltoptions.m4" \
      "${autotoolsdir}/share/aclocal/ltversion.m4" >> aclocal.m4

  echo 'AC_CONFIG_MACRO_DIRS([m4])' >> configure.ac
  sed -i '1s/^/ACLOCAL_AMFLAGS\ \=\ -I\ m4\n/' Makefile.am
  echo 'ACLOCAL_AMFLAGS = -I m4' >> Makefile.am
}

build() {
  cd ${srcdir}/${pkgname}-${pkgver}

  libtoolize && autoheader && aclocal && autoconf && automake --add-missing
  ## If the above fails, use:
  # autoreconf -fvi
  ./configure \
    --prefix=${pkgdir} \
    --enable-intel \
    --enable-fftw --with-fftw=${fftwdir} \
    CC=${CC} FC=${FC} LDFLAGS=${LDFLAGS}
    # --host=x86_64-unknown-linux-gnu \

  make
}

package() {
  set -e
  cd ${srcdir}/${pkgname}-${pkgver}
  # make install
  ## If the above fails, use (with caution):
  make -i install

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
prepare
build
package
