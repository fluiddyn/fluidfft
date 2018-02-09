#!/bin/bash

# pkgdir=$HOME/.local/opt
pkgdir="/cfs/klemming/nobackup/${USER:0:1}/${USER}/opt/pkg"
srcdir=$PWD/autotools
export MAKEFLAGS="-j$(nproc)"

gnu_install() {
  cd $srcdir
  local pkgname=$1
  local pkgver=$2
  local pkgdir=$3

  mkdir -p $pkgdir
  if [ ! -d  ./$pkgname ]
  then
    [ ! -f ${pkgname}-${pkgver}.tar.gz ] && wget ftp://ftp.gnu.org/gnu/${pkgname}/${pkgname}-${pkgver}.tar.gz
    tar xvf ${pkgname}-${pkgver}.tar.gz
    mv ${pkgname}-${pkgver} ${pkgname}

    # git clone git://git.savannah.gnu.org/$pkgname.git --depth=1
  fi

  cd $pkgname
  rm -rf ${pkgdir}
  ./bootstrap
  ./configure --prefix=${pkgdir} CC=gcc
  make
  # make check
  make install
}

package() {
  mkdir -p $srcdir
  local LOCALDIR=$(dirname ${pkgdir})

  # Install GNU stow - a symlink manager: useful to manage local installations
  gnu_install stow "2.2.2" "$LOCALDIR"
  export PATH=$LOCALDIR/bin:$PATH
  rehash
  echo "End of installation: stow"
  sleep 3

  # Install GNU autotools one by one
  for i in autoconf,"2.69" automake,"1.15.1" libtool,"2.4.5"
  do
    pkgname=${i%,*};
    pkgver=${i#*,};
    gnu_install $pkgname $pkgver "$pkgdir/$pkgname"

    cd $pkgdir
    rm $pkgname/share/info/dir
    echo "stow -v $pkgname"
    stow -v $pkgname
    rehash
    echo "End of installation: $pkgname"

    sleep 3

  done

}

# Execute
package
