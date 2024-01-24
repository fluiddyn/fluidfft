# Numpy installation with BLAS and LAPACK

Numpy can potentially perform better when it can make use of
[BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms)
/ [LAPACK](https://en.wikipedia.org/wiki/LAPACK) libraries.
Popular implementations include OpenBLAS, ATLAS and Intel MKL.

Run `fluidinfo -v` to see if your existing `numpy` installation detects any of
the above libraries already provided in the system. If these libraries are
available and `numpy` cannot detect such libraries you may need to configure
the paths into a file located at `$HOME/.numpy-site.cfg` ([See example](https://raw.githubusercontent.com/numpy/numpy/master/site.cfg.example))

## OpenBLAS library

If none of the above libraries are not available, it may be worthwhile to
install it yourself using the following shell script:

```{literalinclude} install_openblas.sh
:language: shell
```
