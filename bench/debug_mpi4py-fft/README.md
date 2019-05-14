
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