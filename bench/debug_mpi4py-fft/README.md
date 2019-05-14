
```
pyenv shell 3.7.2
bash check_segfault.sh
```

- no segfault with pyFFTW==0.10.4

- segfault with pyFFTW==0.11.1 and 0.11.0

- segfault without pyfftw!

## Debug environment

After `bash prepare_env.sh`:

```
. /tmp/tmp_debug/myvenv/bin/activate

# no segfault
cd /tmp/tmp_debug/pyFFTW
hg up 46d5880b5022 && rm -rf build && python setup.py clean && python setup.py install
cd /tmp/tmp_debug/fluidfft
mpirun -np 2 pytest -s

# segfault
cd /tmp/tmp_debug/pyFFTW
hg up 6665eea446f7 && rm -rf build && python setup.py clean && python setup.py install
cd /tmp/tmp_debug/fluidfft
mpirun -np 2 pytest -s

```

```
hg log -G -r 483::484
o  changeset:   484:6665eea446f7
|  user:        Frederik Beaujean <Frederik.Beaujean@lmu.de>
|  date:        Mon May 22 11:56:21 2017 +0200
|  summary:     [setup] Fix link checks
|
@  changeset:   483:46d5880b5022
|  user:        Frederik Beaujean <Frederik.Beaujean@lmu.de>
~  date:        Fri May 19 13:41:20 2017 +0200
   summary:     [setup] minify get_extensions()
```

See https://github.com/pyFFTW/pyFFTW/pull/177/commits/89fc80514333129a92b34af637bcc00a255fff75

The only difference in the installed pyfftw files is in the shared library pyfftw.*.so...

```
ldd pyfftw_*/*.so

pyfftw_segfault/pyfftw.cpython-37m-x86_64-linux-gnu.so:
	linux-vdso.so.1 (0x00007ffd1c5b7000)
	libfftw3q_omp.so.3 => /usr/lib/x86_64-linux-gnu/libfftw3q_omp.so.3 (0x00007f5902274000)
	libfftw3q.so.3 => /usr/lib/x86_64-linux-gnu/libfftw3q.so.3 (0x00007f5901e87000)
	libfftw3l_omp.so.3 => /usr/lib/x86_64-linux-gnu/libfftw3l_omp.so.3 (0x00007f5901c80000)
	libfftw3l.so.3 => /usr/lib/x86_64-linux-gnu/libfftw3l.so.3 (0x00007f59019a2000)
	libfftw3f_omp.so.3 => /usr/lib/x86_64-linux-gnu/libfftw3f_omp.so.3 (0x00007f590179b000)
	libfftw3f.so.3 => /usr/lib/x86_64-linux-gnu/libfftw3f.so.3 (0x00007f590138d000)
	libfftw3_omp.so.3 => /usr/lib/x86_64-linux-gnu/libfftw3_omp.so.3 (0x00007f5901186000)
	libfftw3.so.3 => /usr/lib/x86_64-linux-gnu/libfftw3.so.3 (0x00007f5900d89000)
	...

pyfftw_no_segfault/pyfftw.cpython-37m-x86_64-linux-gnu.so:
	linux-vdso.so.1 (0x00007ffccd11d000)
	libfftw3.so.3 => /usr/lib/x86_64-linux-gnu/libfftw3.so.3 (0x00007f9933042000)
	libfftw3f.so.3 => /usr/lib/x86_64-linux-gnu/libfftw3f.so.3 (0x00007f9932c34000)
	libfftw3l.so.3 => /usr/lib/x86_64-linux-gnu/libfftw3l.so.3 (0x00007f9932956000)
	libfftw3q.so.3 => /usr/lib/x86_64-linux-gnu/libfftw3q.so.3 (0x00007f9932569000)
	libfftw3_threads.so.3 => /usr/lib/x86_64-linux-gnu/libfftw3_threads.so.3 (0x00007f9932362000)
	libfftw3f_threads.so.3 => /usr/lib/x86_64-linux-gnu/libfftw3f_threads.so.3 (0x00007f993215b000)
	libfftw3l_threads.so.3 => /usr/lib/x86_64-linux-gnu/libfftw3l_threads.so.3 (0x00007f9931f54000)
	libfftw3q_threads.so.3 => /usr/lib/x86_64-linux-gnu/libfftw3q_threads.so.3 (0x00007f9931d4d000)
	libfftw3_omp.so.3 => /usr/lib/x86_64-linux-gnu/libfftw3_omp.so.3 (0x00007f9931b46000)
	libfftw3f_omp.so.3 => /usr/lib/x86_64-linux-gnu/libfftw3f_omp.so.3 (0x00007f993193f000)
	libfftw3l_omp.so.3 => /usr/lib/x86_64-linux-gnu/libfftw3l_omp.so.3 (0x00007f9931738000)
	libfftw3q_omp.so.3 => /usr/lib/x86_64-linux-gnu/libfftw3q_omp.so.3 (0x00007f9931531000)
	...
```

This seems to indicate that it is a problem with libfftw3_threads / libfftw3_omp.

```
ldd mpi4py_fft/fftw/fftw_xfftn.cpython-37m-x86_64-linux-gnu.so
	linux-vdso.so.1 (0x00007ffd263e0000)
	libfftw3.so.3 => /usr/lib/x86_64-linux-gnu/libfftw3.so.3 (0x00007fbe64b9f000)
	libfftw3_threads.so.3 => /usr/lib/x86_64-linux-gnu/libfftw3_threads.so.3 (0x00007fbe64998000)
    ...
```

```
ldd fluidfft/fft3d/mpi_with_fftwmpi3d.cpython-37m-x86_64-linux-gnu.so
	linux-vdso.so.1 (0x00007ffe4e5d1000)
	libfftw3.so.3 => /usr/lib/x86_64-linux-gnu/libfftw3.so.3 (0x00007f9a08c10000)
	libfftw3_mpi.so.3 => /usr/lib/x86_64-linux-gnu/libfftw3_mpi.so.3 (0x00007f9a089fa000)
```

```
ldd /usr/lib/x86_64-linux-gnu/libfftw3_mpi.so.3
        linux-vdso.so.1 (0x00007ffcddb1c000)
        libfftw3.so.3 => /usr/lib/x86_64-linux-gnu/libfftw3.so.3 (0x00007f62be0c2000)
        libmpi.so.20 => /usr/lib/x86_64-linux-gnu/libmpi.so.20 (0x00007f62bddcf000)
```
